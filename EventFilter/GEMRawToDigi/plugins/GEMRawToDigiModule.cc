/** \class GEMRawToDigiModule
 *  \unpacker for gem
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include "CondFormats/DataRecord/interface/GEMeMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "CondFormats/GEMObjects/interface/GEMROMapping.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/GEMDigi/interface/AMC13Event.h"
#include "DataFormats/GEMDigi/interface/GEMAMC13EventCollection.h"
#include "DataFormats/GEMDigi/interface/GEMAMCdataCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMGEBdataCollection.h"
#include "DataFormats/GEMDigi/interface/GEMVfatStatusDigiCollection.h"
#include "DataFormats/GEMDigi/interface/VFATdata.h"
#include "EventFilter/GEMRawToDigi/interface/GEMRawToDigi.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class GEMRawToDigiModule : public edm::global::EDProducer<edm::RunCache<GEMROMapping> > {
public:
  /// Constructor
  GEMRawToDigiModule(const edm::ParameterSet& pset);

  // global::EDProducer
  std::shared_ptr<GEMROMapping> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override{};

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<FEDRawDataCollection> fed_token;
  edm::ESGetToken<GEMeMap, GEMeMapRcd> gemEMapToken_;
  bool useDBEMap_;
  bool unPackStatusDigis_;
  std::unique_ptr<GEMRawToDigi> gemRawToDigi_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMRawToDigiModule);

using namespace gem;

GEMRawToDigiModule::GEMRawToDigiModule(const edm::ParameterSet& pset)
    : fed_token(consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("InputLabel"))),
      useDBEMap_(pset.getParameter<bool>("useDBEMap")),
      unPackStatusDigis_(pset.getParameter<bool>("unPackStatusDigis")),
      gemRawToDigi_(std::make_unique<GEMRawToDigi>()) {
  produces<GEMDigiCollection>();
  if (unPackStatusDigis_) {
    produces<GEMVfatStatusDigiCollection>("vfatStatus");
    produces<GEMGEBdataCollection>("gebStatus");
    produces<GEMAMCdataCollection>("AMCdata");
    produces<GEMAMC13EventCollection>("AMC13Event");
  }
  if (useDBEMap_) {
    gemEMapToken_ = esConsumes<GEMeMap, GEMeMapRcd, edm::Transition::BeginRun>();
  }
}

void GEMRawToDigiModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector"));
  desc.add<bool>("useDBEMap", false);
  desc.add<bool>("unPackStatusDigis", false);
  descriptions.add("muonGEMDigisDefault", desc);
}

std::shared_ptr<GEMROMapping> GEMRawToDigiModule::globalBeginRun(edm::Run const&, edm::EventSetup const& iSetup) const {
  auto gemROmap = std::make_shared<GEMROMapping>();
  if (useDBEMap_) {
    const auto& eMap = iSetup.getData(gemEMapToken_);
    auto gemEMap = std::make_unique<GEMeMap>(eMap);
    gemEMap->convert(*gemROmap);
    gemEMap.reset();
  } else {
    // no EMap in DB, using dummy
    auto gemEMap = std::make_unique<GEMeMap>();
    gemEMap->convertDummy(*gemROmap);
    gemEMap.reset();
  }
  return gemROmap;
}

void GEMRawToDigiModule::produce(edm::StreamID iID, edm::Event& iEvent, edm::EventSetup const& iSetup) const {
  auto outGEMDigis = std::make_unique<GEMDigiCollection>();
  auto outVFATStatus = std::make_unique<GEMVfatStatusDigiCollection>();
  auto outGEBStatus = std::make_unique<GEMGEBdataCollection>();
  auto outAMCdata = std::make_unique<GEMAMCdataCollection>();
  auto outAMC13Event = std::make_unique<GEMAMC13EventCollection>();

  // Take raw from the event
  edm::Handle<FEDRawDataCollection> fed_buffers;
  iEvent.getByToken(fed_token, fed_buffers);

  auto gemROMap = runCache(iEvent.getRun().index());

  for (unsigned int fedId = FEDNumbering::MINGEMFEDID; fedId <= FEDNumbering::MAXGEMFEDID; ++fedId) {
    const FEDRawData& fedData = fed_buffers->FEDData(fedId);

    int nWords = fedData.size() / sizeof(uint64_t);
    LogDebug("GEMRawToDigiModule") << "fedId:" << fedId << " words: " << nWords;

    if (nWords < 5)
      continue;

    // trailer checks
    FEDTrailer trailer(fedData.data() + fedData.size() - FEDTrailer::length);

    bool failTrailerCheck = false, failTrailerMatch = false;
    if (!trailer.check() || (trailer.fragmentLength() * sizeof(uint64_t) != fedData.size())) {
      failTrailerCheck = true;
    }

    const unsigned char* data = fedData.data();
    const uint64_t* word = reinterpret_cast<const uint64_t*>(data);
    auto amc13Event = gemRawToDigi_->convertWordToAMC13Event(word);

    if (amc13Event == nullptr) {
      LogDebug("GEMRawToDigiModule") << "AMC13Event FAILED to be produced";
      continue;
    }

    // compare trailers found by last word of fedData.size() and gemRawToDigi
    // caused by error in no. of AMC, GEB or VFAT stored in FEDs
    if ((amc13Event->fragmentLength() != trailer.fragmentLength()) || (amc13Event->crc() != trailer.crc()))
      failTrailerMatch = true;

    LogDebug("GEMRawToDigiModule") << "Event bx:" << iEvent.bunchCrossing() << " lv1Id:" << iEvent.id().event()
                                   << " orbitNumber:" << iEvent.orbitNumber();
    LogDebug("GEMRawToDigiModule") << "AMC13 bx:" << amc13Event->bxId() << " lv1Id:" << int(amc13Event->lv1Id())
                                   << " orbitNumber:" << amc13Event->orbitNumber();

    if (failTrailerCheck || failTrailerMatch) {
      // best to skip these events since FED is most likely corrupt
      edm::LogWarning("GEMRawToDigiModule")
          << "FED trailer: fail check? " << failTrailerCheck << " fail match? " << failTrailerMatch;
    //  continue;
    }

    bool unknownChamber = false, unknownVFat = false, badVfat = false;

    // Read AMC data
    for (auto amcData : *(amc13Event->getAMCpayloads())) {
      uint16_t amcBx = amcData.bx();
      uint8_t amcNum = amcData.amcNum();
      LogDebug("GEMRawToDigiModule") << "AMC no.:" << int(amcData.amcNum()) << " bx:" << int(amcData.bx())
                                     << " lv1Id:" << int(amcData.l1A()) << " orbitNumber:" << int(amcData.orbitNum());

      // Read GEB data
      for (auto gebData : *amcData.gebs()) {
        uint8_t gebId = gebData.inputID();
        GEMROMapping::chamEC geb_ec = {fedId, amcNum, gebId};

        // check if Chamber exists.
        if (!gemROMap->isValidChamber(geb_ec)) {
          unknownChamber = true;
          LogDebug("GEMRawToDigiModule") << "InValid: amcNum " << int(amcNum) << " gebId " << int(gebId);
          continue;
        }

        GEMROMapping::chamDC geb_dc = gemROMap->chamberPos(geb_ec);
        GEMDetId gemChId = geb_dc.detId;

        //Read vfat data
        for (auto vfatData : *gebData.vFATs()) {
          vfatData.setVersion(geb_dc.vfatVer);
          uint16_t vfatId = vfatData.vfatId();
          GEMROMapping::vfatEC vfat_ec = {vfatId, gemChId};

          // check if ChipID exists.
          if (!gemROMap->isValidChipID(vfat_ec)) {
            unknownVFat = true;
            LogDebug("GEMRawToDigiModule") << "InValid: amcNum " << int(amcNum) << " gebId " << int(gebId) << " vfatId "
                                           << int(vfatId) << " vfat Pos " << int(vfatData.position());
            continue;
          }

          // check vfat data
          if (vfatData.quality()) {
            badVfat = true;
            LogDebug("GEMRawToDigiModule")
                << "Quality " << int(vfatData.quality()) << " b1010 " << int(vfatData.b1010()) << " b1100 "
                << int(vfatData.b1100()) << " b1110 " << int(vfatData.b1110());
            if (vfatData.crc() != vfatData.checkCRC()) {
              LogDebug("GEMRawToDigiModule") << "DIFFERENT CRC :" << vfatData.crc() << "   " << vfatData.checkCRC();
            }
          }

          GEMROMapping::vfatDC vfat_dc = gemROMap->vfatPos(vfat_ec);

          vfatData.setPhi(vfat_dc.localPhi);
          GEMDetId gemId = vfat_dc.detId;
          int bx(vfatData.bc() - amcBx);

          for (int chan = 0; chan < VFATdata::nChannels; ++chan) {
            uint8_t chan0xf = 0;
            if (chan < 64)
              chan0xf = ((vfatData.lsData() >> chan) & 0x1);
            else
              chan0xf = ((vfatData.msData() >> (chan - 64)) & 0x1);

            // no hits
            if (chan0xf == 0)
              continue;

            GEMROMapping::channelNum chMap = {vfat_dc.vfatType, chan};
            GEMROMapping::stripNum stMap = gemROMap->hitPos(chMap);

            int stripId = stMap.stNum + vfatData.phi() * GEMeMap::maxChan_;

            GEMDigi digi(stripId, bx);

            LogDebug("GEMRawToDigiModule")
                << " fed: " << fedId << " amc:" << int(amcNum) << " geb:" << int(gebId) << " vfat:" << vfat_dc.localPhi
                << ",type: " << vfat_dc.vfatType << " id:" << gemId << " ch:" << chMap.chNum << " st:" << digi.strip()
                << " bx:" << digi.bx();

            outGEMDigis.get()->insertDigi(gemId, digi);

          }  // end of channel loop

          if (unPackStatusDigis_) {
            outVFATStatus.get()->insertDigi(gemId, GEMVfatStatusDigi(vfatData));
          }

        }  // end of vfat loop

        if (unPackStatusDigis_) {
          gebData.clearVFATs();
          outGEBStatus.get()->insertDigi(gemChId.chamberId(), (gebData));
        }

      }  // end of geb loop

      if (unPackStatusDigis_) {
        amcData.clearGEBs();
        outAMCdata.get()->insertDigi(amcData.boardId(), (amcData));
      }

    }  // end of amc loop

    if (unPackStatusDigis_) {
      amc13Event->clearAMCpayloads();
      outAMC13Event.get()->insertDigi(amc13Event->bxId(), AMC13Event(*amc13Event));
    }

    if (unknownChamber || unknownVFat || badVfat) {
      edm::LogWarning("GEMRawToDigiModule") << "unpacking error: unknown Chamber " << unknownChamber << " unknown VFat "
                                            << unknownVFat << " bad VFat " << badVfat;
    }

  }  // end of amc13Event

  iEvent.put(std::move(outGEMDigis));

  if (unPackStatusDigis_) {
    iEvent.put(std::move(outVFATStatus), "vfatStatus");
    iEvent.put(std::move(outGEBStatus), "gebStatus");
    iEvent.put(std::move(outAMCdata), "AMCdata");
    iEvent.put(std::move(outAMC13Event), "AMC13Event");
  }
}
