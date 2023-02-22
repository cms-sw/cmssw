#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "EventFilter/HGCalRawToDigi/interface/HGCalUnpacker.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiCollections.h"

class HGCalRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit HGCalRawToDigi(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<FEDRawDataCollection> fedRawToken_;
  edm::EDPutTokenT<HGCalDigiCollection> digisToken_;
  edm::EDPutTokenT<HGCalElecDigiCollection> elecDigisToken_;

  const std::vector<unsigned int> fedIds_;
  std::unique_ptr<HGCalUnpacker<HGCalElectronicsId> > unpacker_;
};

HGCalRawToDigi::HGCalRawToDigi(const edm::ParameterSet& iConfig)
    : fedRawToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      digisToken_(produces<HGCalDigiCollection>()),
      elecDigisToken_(produces<HGCalElecDigiCollection>()),
      fedIds_(iConfig.getParameter<std::vector<unsigned int> >("fedIds")),
      unpacker_(new HGCalUnpacker<HGCalElectronicsId>(
          HGCalUnpackerConfig{.sLinkBOE = iConfig.getParameter<unsigned int>("slinkBOE"),
                              .captureBlockReserved = iConfig.getParameter<unsigned int>("captureBlockReserved"),
                              .econdHeaderMarker = iConfig.getParameter<unsigned int>("econdHeaderMarker"),
                              .sLinkCaptureBlockMax = iConfig.getParameter<unsigned int>("maxCaptureBlock"),
                              .captureBlockECONDMax = iConfig.getParameter<unsigned int>("captureBlockECONDMax"),
                              .econdERXMax = iConfig.getParameter<unsigned int>("econdERXMax"),
                              .erxChannelMax = iConfig.getParameter<unsigned int>("erxChannelMax"),
                              .payloadLengthMax = iConfig.getParameter<unsigned int>("payloadLengthMax"),
                              .channelMax = iConfig.getParameter<unsigned int>("channelMax"),
                              .commonModeMax = iConfig.getParameter<unsigned int>("commonModeMax"),
                              .badECONDMax = iConfig.getParameter<unsigned int>("badECONDMax")})) {}

void HGCalRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // retrieve the FED raw data
  const auto& raw_data = iEvent.get(fedRawToken_);
  // prepare the output
  HGCalDigiCollection digis;
  HGCalElecDigiCollection elec_digis;
  for (const auto& fed_id : fedIds_) {
    const auto& fed_data = raw_data.FEDData(fed_id);
    if (fed_data.size() == 0)
      continue;  //FIXME reporting?

    //FIXME better way to recast char's to uint32_t's?
    std::vector<uint32_t> uint_data;
    auto* ptr = fed_data.data();
    for (size_t i = 0; i < fed_data.size(); i += 4)
      uint_data.emplace_back(((*(ptr + i) & 0xff) << 0) + ((*(ptr + i + 1) & 0xff) << 8) +
                             ((*(ptr + i + 2) & 0xff) << 16) + ((*(ptr + i + 3) & 0xff) << 24));
    //FIXME test if we are at the end of the buffer

    unpacker_->parseSLink(
        uint_data.data(),
        uint_data.size(),
        [](uint16_t sLink, uint8_t captureBlock, uint8_t econd) -> uint16_t {
          if (sLink == 0 && captureBlock == 0 && econd == 3)
            return 0b1;
          return 0b11;
        },
        [](HGCalElectronicsId elecID) -> HGCalElectronicsId { return elecID; });
    auto elecid_to_detid = [](const HGCalElectronicsId& id) -> HGCalDetId {
      return HGCalDetId(id.raw());  //FIXME not at all!!
    };
    //FIXME replace lambda's by something more relevant?

    auto channeldata = unpacker_->getChannelData();
    auto cms = unpacker_->getCommonModeIndex();
    for (unsigned int i = 0; i < channeldata.size(); i++) {
      auto data = channeldata.at(i);
      auto cm = cms.at(i);
      auto id = data.id();
      auto idraw = id.raw();
      auto raw = data.raw();
      edm::LogWarning("HGCalRawToDigi:produce") << "id=" << idraw << ", raw=" << raw << ", common mode index=" << cm;
      digis.push_back(
          HGCROCChannelDataFrameSpec(elecid_to_detid(data.id()), data.raw()));  //FIXME to be checked by Yulun.
      elec_digis.push_back(data);
    }
    if (const auto& bad_econds = unpacker_->getBadECOND(); !bad_econds.empty())
      edm::LogWarning("HGCalRawToDigi:produce").log([&bad_econds](auto& log) {
        log << "Bad ECON-D: " << std::dec;
        std::string prefix;
        for (const auto& badECOND : bad_econds)
          log << prefix << badECOND, prefix = ", ";
        log << ".";
      });
  }
  iEvent.emplace(digisToken_, std::move(digis));
  iEvent.emplace(elecDigisToken_, std::move(elec_digis));
}

void HGCalRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("rawDataCollector"));
  desc.add<unsigned int>("maxCaptureBlock", 1)->setComment("maximum number of capture blocks in one S-Link");
  desc.add<unsigned int>("captureBlockReserved", 0)->setComment("capture block reserved pattern");
  desc.add<unsigned int>("econdHeaderMarker", 0x154)->setComment("ECON-D header Marker patter");
  desc.add<unsigned int>("slinkBOE", 0x2a)->setComment("SLink BOE pattern");
  desc.add<unsigned int>("captureBlockECONDMax", 12)->setComment("maximum number of ECON-D's in one capture block");
  desc.add<unsigned int>("econdERXMax", 12)->setComment("maximum number of eRx's in one ECON-D");
  desc.add<unsigned int>("erxChannelMax", 37)->setComment("maximum number of channels in one eRx");
  desc.add<unsigned int>("payloadLengthMax", 469)->setComment("maximum length of payload length");
  desc.add<unsigned int>("channelMax", 7000000)->setComment("maximum number of channels unpacked");
  desc.add<unsigned int>("commonModeMax", 4000000)->setComment("maximum number of common modes unpacked");
  desc.add<unsigned int>("badECONDMax", 200)->setComment("maximum number of bad ECON-D's");
  desc.add<std::vector<unsigned int> >("fedIds", {});
  descriptions.add("hgcalDigis", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalRawToDigi);
