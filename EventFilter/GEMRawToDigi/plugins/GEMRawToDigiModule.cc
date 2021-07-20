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
#include "DataFormats/GEMDigi/interface/GEMAMC13StatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMAMCStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMOHStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMVFATStatusCollection.h"
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
  bool useDBEMap_, keepDAQStatus_, readMultiBX_;
  bool unPackStatusDigis_;
  std::unique_ptr<GEMRawToDigi> gemRawToDigi_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMRawToDigiModule);

using namespace gem;

GEMRawToDigiModule::GEMRawToDigiModule(const edm::ParameterSet& pset)
    : fed_token(consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("InputLabel"))),
      useDBEMap_(pset.getParameter<bool>("useDBEMap")),
      keepDAQStatus_(pset.getParameter<bool>("keepDAQStatus")),
      readMultiBX_(pset.getParameter<bool>("readMultiBX")),
      unPackStatusDigis_(pset.getParameter<bool>("unPackStatusDigis")),
      gemRawToDigi_(std::make_unique<GEMRawToDigi>()) {
  produces<GEMDigiCollection>();
  if (keepDAQStatus_) {
    produces<GEMAMC13StatusCollection>("AMC13Status");
    produces<GEMAMCStatusCollection>("AMCStatus");
    produces<GEMOHStatusCollection>("OHStatus");
    produces<GEMVFATStatusCollection>("VFATStatus");
  }
  if (unPackStatusDigis_) {
    // to be removed
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
  desc.add<bool>("keepDAQStatus", false);
  desc.add<bool>("readMultiBX", false);
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
  auto outAMC13Status = std::make_unique<GEMAMC13StatusCollection>();
  auto outAMCStatus = std::make_unique<GEMAMCStatusCollection>();
  auto outOHStatus = std::make_unique<GEMOHStatusCollection>();
  auto outVFATStatus = std::make_unique<GEMVFATStatusCollection>();

  // to be removed
  auto outVFATStatusOld = std::make_unique<GEMVfatStatusDigiCollection>();
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
    GEMAMC13Status st_amc13(fedData);
    if (st_amc13.isBad()) {
      LogDebug("GEMRawToDigiModule") << st_amc13;
      if (keepDAQStatus_) {
        outAMC13Status.get()->insertDigi(fedId, st_amc13);
      }
      continue;
    }

    const uint64_t* word = reinterpret_cast<const uint64_t*>(fedData.data());
    auto amc13Event = gemRawToDigi_->convertWordToAMC13Event(word);
    LogDebug("GEMRawToDigiModule") << "Event bx:" << iEvent.bunchCrossing() << " lv1Id:" << iEvent.id().event()
                                   << " orbitNumber:" << iEvent.orbitNumber();
    LogDebug("GEMRawToDigiModule") << "AMC13 bx:" << amc13Event->bunchCrossing()
                                   << " lv1Id:" << int(amc13Event->lv1Id())
                                   << " orbitNumber:" << amc13Event->orbitNumber();

    // Read AMC data
    for (auto amcData : *(amc13Event->getAMCpayloads())) {
      uint8_t amcNum = amcData.amcNum();
      GEMROMapping::sectorEC amcEC{fedId, amcNum};
      if (!gemROMap->isValidAMC(amcEC)) {
        st_amc13.inValidAMC();
        continue;
      }

      GEMAMCStatus st_amc(amc13Event.get(), amcData);
      if (st_amc.isBad()) {
        LogDebug("GEMRawToDigiModule") << st_amc;
        if (keepDAQStatus_) {
          outAMCStatus.get()->insertDigi(fedId, st_amc);
        }
        continue;
      }

      uint16_t amcBx = amcData.bunchCrossing();
      LogDebug("GEMRawToDigiModule") << "AMC no.:" << int(amcData.amcNum()) << " bx:" << int(amcData.bunchCrossing())
                                     << " lv1Id:" << int(amcData.lv1Id())
                                     << " orbitNumber:" << int(amcData.orbitNumber());

      // Read GEB data
      for (auto optoHybrid : *amcData.gebs()) {
        uint8_t gebId = optoHybrid.inputID();
        GEMROMapping::chamEC geb_ec{fedId, amcNum, gebId};

        bool isValidChamber = gemROMap->isValidChamber(geb_ec);
        if (!isValidChamber) {
          st_amc.inValidOH();
          continue;
        }
        GEMROMapping::chamDC geb_dc = gemROMap->chamberPos(geb_ec);
        GEMDetId gemChId = geb_dc.detId;

        GEMOHStatus st_oh(optoHybrid);
        if (st_oh.isBad()) {
          LogDebug("GEMRawToDigiModule") << st_oh;
          if (keepDAQStatus_) {
            outOHStatus.get()->insertDigi(gemChId, st_oh);
          }
        }

        //Read vfat data
        for (auto vfatData : *optoHybrid.vFATs()) {
          // set vfat fw version
          vfatData.setVersion(geb_dc.vfatVer);
          uint16_t vfatId = vfatData.vfatId();
          GEMROMapping::vfatEC vfat_ec{vfatId, gemChId};

          if (!gemROMap->isValidChipID(vfat_ec)) {
            st_oh.inValidVFAT();
            continue;
          }

          GEMROMapping::vfatDC vfat_dc = gemROMap->vfatPos(vfat_ec);
          vfatData.setPhi(vfat_dc.localPhi);
          GEMDetId gemId = vfat_dc.detId;

          GEMVFATStatus st_vfat(amcData, vfatData, vfatData.phi(), readMultiBX_);
          if (st_vfat.isBad()) {
            LogDebug("GEMRawToDigiModule") << st_vfat;
            if (keepDAQStatus_) {
              outVFATStatus.get()->insertDigi(gemId, st_vfat);
            }
            continue;
          }

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

            GEMROMapping::channelNum chMap{vfat_dc.vfatType, chan};
            GEMROMapping::stripNum stMap = gemROMap->hitPos(chMap);

            int stripId = stMap.stNum + vfatData.phi() * GEMeMap::maxChan_;

            GEMDigi digi(stripId, bx);

            LogDebug("GEMRawToDigiModule")
                << "fed: " << fedId << " amc:" << int(amcNum) << " geb:" << int(gebId) << " vfat id:" << int(vfatId)
                << ",type:" << vfat_dc.vfatType << " id:" << gemId << " ch:" << chMap.chNum << " st:" << digi.strip()
                << " bx:" << digi.bx();

            outGEMDigis.get()->insertDigi(gemId, digi);

          }  // end of channel loop

          if (keepDAQStatus_) {
            outVFATStatus.get()->insertDigi(gemId, st_vfat);
          }
          if (unPackStatusDigis_) {
            outVFATStatusOld.get()->insertDigi(gemId, GEMVfatStatusDigi(vfatData));
          }

        }  // end of vfat loop

        if (keepDAQStatus_) {
          outOHStatus.get()->insertDigi(gemChId, st_oh);
        }
        if (unPackStatusDigis_) {
          optoHybrid.clearVFATs();
          outGEBStatus.get()->insertDigi(gemChId.chamberId(), (optoHybrid));
        }

      }  // end of optohybrid loop

      if (keepDAQStatus_) {
        outAMCStatus.get()->insertDigi(fedId, st_amc);
      }
      if (unPackStatusDigis_) {
        amcData.clearGEBs();
        outAMCdata.get()->insertDigi(amcData.boardId(), (amcData));
      }

    }  // end of amc loop

    if (keepDAQStatus_) {
      outAMC13Status.get()->insertDigi(fedId, st_amc13);
    }

    if (unPackStatusDigis_) {
      amc13Event->clearAMCpayloads();
    }

  }  // end of amc13Event

  iEvent.put(std::move(outGEMDigis));

  if (keepDAQStatus_) {
    iEvent.put(std::move(outAMC13Status), "AMC13Status");
    iEvent.put(std::move(outAMCStatus), "AMCStatus");
    iEvent.put(std::move(outOHStatus), "OHStatus");
    iEvent.put(std::move(outVFATStatus), "VFATStatus");
  }
  if (unPackStatusDigis_) {
    // to be removed
    iEvent.put(std::move(outVFATStatusOld), "vfatStatus");
    iEvent.put(std::move(outGEBStatus), "gebStatus");
    iEvent.put(std::move(outAMCdata), "AMCdata");
    iEvent.put(std::move(outAMC13Event), "AMC13Event");
  }
}
