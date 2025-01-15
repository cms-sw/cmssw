// -*- C++ -*-
//

#include <utility>
#include <unordered_map>

#include <string>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"

#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITChipBitStream.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITDTCCollection.h"

class Phase2TrackerDTCAssociator : public edm::one::EDProducer<> {
public:
  explicit Phase2TrackerDTCAssociator(const edm::ParameterSet&);
  ~Phase2TrackerDTCAssociator() override;

private:
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
  const edm::EDGetTokenT<edm::DetSetVector<Phase2ITChipBitStream>> ITChipBitStreamToken_;

  void AddHexToPtr(unsigned char *data_ptr, int word_index, const std::vector<bool>& moduleBitStream);

};

Phase2TrackerDTCAssociator::Phase2TrackerDTCAssociator(const edm::ParameterSet& iConfig)
  : cablingMapToken_(esConsumes()),
    ITChipBitStreamToken_(consumes<edm::DetSetVector<Phase2ITChipBitStream>>(iConfig.getParameter<edm::InputTag>("Phase2ITChipBitStream"))){
    produces<FEDRawDataCollection>();
}

Phase2TrackerDTCAssociator::~Phase2TrackerDTCAssociator() {}

void Phase2TrackerDTCAssociator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    using namespace edm;
    using namespace std;
    static const int MIN_DTC_ID = 0;
    static const int MAX_DTC_ID = 36;
    static const int MIN_SLINK_ID = 0;
    static const int MAX_SLINK_ID = 15;
    static const int SLINKS_PER_DTC = 16;

    unsigned int EventID = iEvent.id().event();
    const auto& cablingMap = iSetup.getData(cablingMapToken_);
    auto fedRawDataCollection = std::make_unique<FEDRawDataCollection>();

    // loop over modules
    for (const auto& detSet : iEvent.get(ITChipBitStreamToken_)){
        // FIXME ignore header and trailer for now

        DetId detId = detSet.detId();
        auto DTCELinkId = cablingMap.detIdToDTCELinkId(detId);
        int dtc_id = (*DTCELinkId.first).second.dtc_id();
        unsigned int slinkIndex = dtc_id; // FIXME for now just dumping data stream to one DTC to one slink
        // detId.rawId() : can do this but need to loop over again
        // Define a map file, xml or db to do the remapping

        std::vector<bool> offset_bits;
        std::vector<bool> data_bits;
        unsigned int bitstream_cumulative = 0;

        // loop over chips for a given module
        for (const auto& chip : detSet){
            std::vector<bool> bitstream = chip.get_bitstream();

            // 16 bit configuration but offset is split into MSB and LSB
            unsigned int offset_chip = bitstream_cumulative/32;

            // MSB LSB dummy value
            std::vector<bool> offset_MSB(16, false);
            std::vector<bool> offset_LSB(16, false);
            for (unsigned int i=0; i<16; ++i){
                offset_MSB[15-i] = (offset_chip >> (i+16)) & 1;
                offset_LSB[15-i] = (offset_chip >> i) & 1;
            }
            offset_bits.insert(offset_bits.end(), offset_MSB.begin(), offset_MSB.end());
            offset_bits.insert(offset_bits.end(), offset_LSB.begin(), offset_LSB.end());

            unsigned int padding_offset_bits = (128 - (offset_bits.size() % 128)) % 128;
            if (padding_offset_bits > 0){
                offset_bits.insert(offset_bits.end(), padding_offset_bits, false);
            }

            std::vector<bool> data_header(16, false); // inject dummy
            std::vector<bool> data_chipsize(16, false); // inject dummy
            data_bits.insert(data_bits.end(), data_header.begin(), data_header.end());
            data_bits.insert(data_bits.end(), data_chipsize.begin(), data_chipsize.end());
            data_bits.insert(data_bits.end(), bitstream.begin(), bitstream.end());
            
            unsigned int padding_data_bits = (128 - (data_bits.size() % 128)) % 128;

            if (padding_data_bits > 0){
                data_bits.insert(data_bits.end(), padding_data_bits, false);
            }

            bitstream_cumulative += data_bits.size();
        }

        unsigned int offset_bytes = (offset_bits.size() + 7) / 8;
        unsigned int data_bytes = (data_bits.size() + 7) / 8;

        FEDRawData combined_slink;
        combined_slink.resize(offset_bytes + data_bytes);
        unsigned char *ptr = combined_slink.data();

        for (unsigned int i = 0; i < offset_bits.size() / 16; ++i) {
            AddHexToPtr(ptr, i, offset_bits);
        }
        for (unsigned int i = 0; i < data_bits.size() / 16; ++i) {
            AddHexToPtr(ptr, i + offset_bits.size() / 16, data_bits);
        }

        FEDRawData& current_slink = fedRawDataCollection->FEDData(slinkIndex);
        unsigned int current_slink_size = current_slink.size();
        unsigned int new_slink_size = current_slink_size + combined_slink.size();
        current_slink.resize(new_slink_size);
        std::memcpy(current_slink.data() + current_slink_size, combined_slink.data(), combined_slink.size());

    }


    //for (int dtc_id = MIN_DTC_ID; dtc_id < MAX_DTC_ID; dtc_id++){
    //    for (int slink_id = 0; slink_id < MAX_SLINK_ID + 1; slink_id++){
    //    }
    //}
}

void Phase2TrackerDTCAssociator::AddHexToPtr(unsigned char *ptr, int index, const std::vector<bool>& bits)
{
    uint16_t hex_word = 0;

    for (int i = 0; i < 16; ++i) {
        if (bits[index * 16 + i]) {
            hex_word |= (1 << (15 - i));
        }
    }

    ptr[index * 4 + 0] = (hex_word >> 12) & 0xF;
    ptr[index * 4 + 1] = (hex_word >> 8) & 0xF;
    ptr[index * 4 + 2] = (hex_word >> 4) & 0xF;
    ptr[index * 4 + 3] = (hex_word >> 0) & 0xF;
}

void Phase2TrackerDTCAssociator::beginJob() {}

void Phase2TrackerDTCAssociator::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2TrackerDTCAssociator);


