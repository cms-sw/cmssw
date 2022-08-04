/** \class GEMRawToDigiModule
 *  \unpacker for gem
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include "CondFormats/DataRecord/interface/GEMChMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMChMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/GEMDigi/interface/GEMAMC13StatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMAMCStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMOHStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMVFATStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
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

class GEMRawToDigiModule : public edm::global::EDProducer<edm::RunCache<GEMChMap>> {
public:
  /// Constructor
  GEMRawToDigiModule(const edm::ParameterSet& pset);

  // global::EDProducer
  std::shared_ptr<GEMChMap> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override{};

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<FEDRawDataCollection> fed_token;
  edm::ESGetToken<GEMChMap, GEMChMapRcd> gemChMapToken_;
  bool useDBEMap_, keepDAQStatus_, readMultiBX_, ge21Off_;
  unsigned int fedIdStart_, fedIdEnd_;
  std::unique_ptr<GEMRawToDigi> gemRawToDigi_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMRawToDigiModule);

GEMRawToDigiModule::GEMRawToDigiModule(const edm::ParameterSet& pset)
    : fed_token(consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("InputLabel"))),
      useDBEMap_(pset.getParameter<bool>("useDBEMap")),
      keepDAQStatus_(pset.getParameter<bool>("keepDAQStatus")),
      readMultiBX_(pset.getParameter<bool>("readMultiBX")),
      ge21Off_(pset.getParameter<bool>("ge21Off")),
      fedIdStart_(pset.getParameter<unsigned int>("fedIdStart")),
      fedIdEnd_(pset.getParameter<unsigned int>("fedIdEnd")),
      gemRawToDigi_(std::make_unique<GEMRawToDigi>()) {
  produces<GEMDigiCollection>();
  if (keepDAQStatus_) {
    produces<GEMAMC13StatusCollection>("AMC13Status");
    produces<GEMAMCStatusCollection>("AMCStatus");
    produces<GEMOHStatusCollection>("OHStatus");
    produces<GEMVFATStatusCollection>("VFATStatus");
  }
  if (useDBEMap_) {
    gemChMapToken_ = esConsumes<GEMChMap, GEMChMapRcd, edm::Transition::BeginRun>();
  }
  if (ge21Off_ && fedIdStart_ == FEDNumbering::MINGEMFEDID && fedIdEnd_ == FEDNumbering::MAXGEMFEDID) {
    fedIdEnd_ = FEDNumbering::MINGE21FEDID - 1;
  } else if (ge21Off_) {
    edm::LogError("InvalidSettings") << "Turning GE2/1 off requires changing the FEDIDs the GEM unpacker looks at. If "
                                        "you wish to set the FEDIDs yourself, don't use the ge21Off switch.";
  }
}

void GEMRawToDigiModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector"));
  desc.add<bool>("useDBEMap", false);
  desc.add<bool>("keepDAQStatus", true);
  desc.add<bool>("readMultiBX", false);
  desc.add<bool>("ge21Off", false);
  desc.add<unsigned int>("fedIdStart", FEDNumbering::MINGEMFEDID);
  desc.add<unsigned int>("fedIdEnd", FEDNumbering::MAXGEMFEDID);
  descriptions.add("muonGEMDigisDefault", desc);
}

std::shared_ptr<GEMChMap> GEMRawToDigiModule::globalBeginRun(edm::Run const&, edm::EventSetup const& iSetup) const {
  if (useDBEMap_) {
    const auto& eMap = iSetup.getData(gemChMapToken_);
    auto gemChMap = std::make_shared<GEMChMap>(eMap);
    return gemChMap;
  } else {
    // no EMap in DB, using dummy
    auto gemChMap = std::make_shared<GEMChMap>();
    gemChMap->setDummy();
    return gemChMap;
  }
}

void GEMRawToDigiModule::produce(edm::StreamID iID, edm::Event& iEvent, edm::EventSetup const& iSetup) const {
  auto outGEMDigis = std::make_unique<GEMDigiCollection>();
  auto outAMC13Status = std::make_unique<GEMAMC13StatusCollection>();
  auto outAMCStatus = std::make_unique<GEMAMCStatusCollection>();
  auto outOHStatus = std::make_unique<GEMOHStatusCollection>();
  auto outVFATStatus = std::make_unique<GEMVFATStatusCollection>();

  // Take raw from the event
  edm::Handle<FEDRawDataCollection> fed_buffers;
  iEvent.getByToken(fed_token, fed_buffers);

  auto gemChMap = runCache(iEvent.getRun().index());

  for (unsigned int fedId = fedIdStart_; fedId <= fedIdEnd_; ++fedId) {
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
    auto amc13 = gemRawToDigi_->convertWordToGEMAMC13(word);
    LogDebug("GEMRawToDigiModule") << "Event bx:" << iEvent.bunchCrossing() << " lv1Id:" << iEvent.id().event()
                                   << " orbitNumber:" << iEvent.orbitNumber();
    LogDebug("GEMRawToDigiModule") << "AMC13 bx:" << amc13->bunchCrossing() << " lv1Id:" << int(amc13->lv1Id())
                                   << " orbitNumber:" << amc13->orbitNumber();

    // Read AMC data
    for (const auto& amc : *(amc13->getAMCpayloads())) {
      uint8_t amcNum = amc.amcNum();
      if (!gemChMap->isValidAMC(fedId, amcNum)) {
        st_amc13.inValidAMC();
        continue;
      }

      GEMAMCStatus st_amc(amc13.get(), amc);
      if (st_amc.isBad()) {
        LogDebug("GEMRawToDigiModule") << st_amc;
        if (keepDAQStatus_) {
          outAMCStatus.get()->insertDigi(fedId, st_amc);
        }
        continue;
      }

      uint16_t amcBx = amc.bunchCrossing();
      LogDebug("GEMRawToDigiModule") << "AMC no.:" << int(amc.amcNum()) << " bx:" << int(amc.bunchCrossing())
                                     << " lv1Id:" << int(amc.lv1Id()) << " orbitNumber:" << int(amc.orbitNumber());

      // Read GEB data
      for (const auto& optoHybrid : *amc.gebs()) {
        uint8_t gebId = optoHybrid.inputID();

        bool isValidChamber = gemChMap->isValidChamber(fedId, amcNum, gebId);
        if (!isValidChamber) {
          st_amc.inValidOH();
          continue;
        }
        auto geb_dc = gemChMap->chamberPos(fedId, amcNum, gebId);
        GEMDetId cId(geb_dc.detId);
        int chamberType = geb_dc.chamberType;

        GEMOHStatus st_oh(optoHybrid);
        if (st_oh.isBad()) {
          LogDebug("GEMRawToDigiModule") << st_oh;
          if (keepDAQStatus_) {
            outOHStatus.get()->insertDigi(cId, st_oh);
          }
        }

        //Read vfat data
        for (auto vfat : *optoHybrid.vFATs()) {
          // set vfat fw version
          if (chamberType < 10)
            vfat.setVersion(2);
          else
            vfat.setVersion(3);
          uint16_t vfatId = vfat.vfatId();

          if (!gemChMap->isValidVFAT(chamberType, vfatId)) {
            st_oh.inValidVFAT();
            continue;
          }

          GEMVFATStatus st_vfat(amc, vfat, vfat.phi(), readMultiBX_);
          if (st_vfat.isBad()) {
            LogDebug("GEMRawToDigiModule") << st_vfat;
            if (keepDAQStatus_) {
              outVFATStatus.get()->insertDigi(cId, st_vfat);
            }
            continue;
          }

          int bx(vfat.bc() - amcBx);

          for (int chan = 0; chan < GEMVFAT::nChannels; ++chan) {
            uint8_t chan0xf = 0;
            if (chan < 64)
              chan0xf = ((vfat.lsData() >> chan) & 0x1);
            else
              chan0xf = ((vfat.msData() >> (chan - 64)) & 0x1);

            // no hits
            if (chan0xf == 0)
              continue;

            auto stMap = gemChMap->getStrip(chamberType, vfatId, chan);

            int stripId = stMap.stNum;
            int ieta = stMap.iEta;

            GEMDetId gemId(cId.region(), cId.ring(), cId.station(), cId.layer(), cId.chamber(), ieta);

            GEMDigi digi(stripId, bx);

            LogDebug("GEMRawToDigiModule") << "fed: " << fedId << " amc:" << int(amcNum) << " geb:" << int(gebId)
                                           << " vfat id:" << int(vfatId) << ",type:" << chamberType << " id:" << gemId
                                           << " ch:" << chan << " st:" << digi.strip() << " bx:" << digi.bx();

            outGEMDigis.get()->insertDigi(gemId, digi);

          }  // end of channel loop

          if (keepDAQStatus_) {
            outVFATStatus.get()->insertDigi(cId, st_vfat);
          }

        }  // end of vfat loop

        if (keepDAQStatus_) {
          outOHStatus.get()->insertDigi(cId, st_oh);
        }

      }  // end of optohybrid loop

      if (keepDAQStatus_) {
        outAMCStatus.get()->insertDigi(fedId, st_amc);
      }

    }  // end of amc loop

    if (keepDAQStatus_) {
      outAMC13Status.get()->insertDigi(fedId, st_amc13);
    }

  }  // end of amc13

  iEvent.put(std::move(outGEMDigis));

  if (keepDAQStatus_) {
    iEvent.put(std::move(outAMC13Status), "AMC13Status");
    iEvent.put(std::move(outAMCStatus), "AMCStatus");
    iEvent.put(std::move(outOHStatus), "OHStatus");
    iEvent.put(std::move(outVFATStatus), "VFATStatus");
  }
}
