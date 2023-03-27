#include "DQM/EcalMonitorTasks/interface/OccupancyTask.h"
#include "FWCore/Framework/interface/Event.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"

namespace ecaldqm {
  OccupancyTask::OccupancyTask() : DQWorkerTask(), recHitThreshold_(0.), tpThreshold_(0.), m_iTime(0.) {}

  void OccupancyTask::setParams(edm::ParameterSet const& _params) {
    recHitThreshold_ = _params.getUntrackedParameter<double>("recHitThreshold");
    tpThreshold_ = _params.getUntrackedParameter<double>("tpThreshold");
    metadataTag = _params.getParameter<edm::InputTag>("metadata");
    lumiCheck_ = _params.getUntrackedParameter<bool>("lumiCheck", false);
    if (!onlineMode_) {
      MEs_.erase(std::string("PU"));
      MEs_.erase(std::string("NEvents"));
      MEs_.erase(std::string("TrendEventsperLumi"));
      MEs_.erase(std::string("TrendPUperLumi"));
      MEs_.erase(std::string("AELoss"));
      MEs_.erase(std::string("AEReco"));
    }
  }

  void OccupancyTask::setTokens(edm::ConsumesCollector& _collector) {
    lasertoken_ = _collector.esConsumes();
    metaDataToken_ = _collector.consumes<OnlineLuminosityRecord>(metadataTag);
  }

  bool OccupancyTask::filterRunType(short const* _runType) {
    for (int iFED(0); iFED < 54; iFED++) {
      if (_runType[iFED] == EcalDCCHeaderBlock::COSMIC || _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
          _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
          _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL || _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
          _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL)
        return true;
    }

    return false;
  }

  void OccupancyTask::beginRun(edm::Run const&, edm::EventSetup const& _es) { FillLaser = true; }
  void OccupancyTask::beginEvent(edm::Event const& _evt,
                                 edm::EventSetup const& _es,
                                 bool const& ByLumiResetSwitch,
                                 bool&) {
    if (ByLumiResetSwitch) {
      MEs_.at("DigiAllByLumi").reset(GetElectronicsMap());
      MEs_.at("TPDigiThrAllByLumi").reset(GetElectronicsMap());
      MEs_.at("RecHitThrAllByLumi").reset(GetElectronicsMap());
      nEv = 0;
      if (onlineMode_) {
        MEs_.at("PU").reset(GetElectronicsMap(), -1);
        MEs_.at("NEvents").reset(GetElectronicsMap(), -1);
        FindPUinLS = true;
      }
    }
    nEv++;
    MESet& meLaserCorrProjEta(MEs_.at("LaserCorrProjEta"));
    m_iTime = _evt.time().value();
    if (FillLaser) {
      float lasercalib = 1.;
      auto const& laser = &_es.getData(lasertoken_);
      const edm::Timestamp& evtTimeStamp = edm::Timestamp(m_iTime);

      for (int i = 0; i < EBDetId::kSizeForDenseIndexing; i++) {
        if (!EBDetId::validDenseIndex(i))
          continue;
        EBDetId ebid(EBDetId::unhashIndex(i));
        lasercalib = laser->getLaserCorrection(ebid, evtTimeStamp);
        meLaserCorrProjEta.fill(getEcalDQMSetupObjects(), ebid, lasercalib);
      }

      for (int i = 0; i < EEDetId::kSizeForDenseIndexing; i++) {
        if (!EEDetId::validDenseIndex(i))
          continue;
        EEDetId eeid(EEDetId::unhashIndex(i));
        lasercalib = laser->getLaserCorrection(eeid, evtTimeStamp);
        meLaserCorrProjEta.fill(getEcalDQMSetupObjects(), eeid, lasercalib);
      }
      FillLaser = false;
    }
    if (lumiCheck_ && FindPUinLS) {
      scal_pu = -1.;
      MESet& mePU(static_cast<MESet&>(MEs_.at("PU")));
      edm::Handle<OnlineLuminosityRecord> metaData;
      _evt.getByToken(metaDataToken_, metaData);

      if (metaData.isValid())
        scal_pu = metaData->avgPileUp();
      mePU.fill(getEcalDQMSetupObjects(), double(scal_pu));
      FindPUinLS = false;
    }
  }

  void OccupancyTask::runOnRawData(EcalRawDataCollection const& _dcchs) {
    MESet& meDCC(MEs_.at("DCC"));

    for (EcalRawDataCollection::const_iterator dcchItr(_dcchs.begin()); dcchItr != _dcchs.end(); ++dcchItr)
      meDCC.fill(getEcalDQMSetupObjects(), dcchItr->id());
  }

  void OccupancyTask::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
    if (onlineMode_) {
      MESet& meNEvents(static_cast<MESet&>(MEs_.at("NEvents")));
      MESet& meTrendEventsperLumi(MEs_.at("TrendEventsperLumi"));
      MESet& meTrendPUperLumi(MEs_.at("TrendPUperLumi"));

      meNEvents.fill(getEcalDQMSetupObjects(), double(nEv));
      meTrendEventsperLumi.fill(getEcalDQMSetupObjects(), EcalBarrel, double(timestamp_.iLumi), double(nEv));
      meTrendPUperLumi.fill(getEcalDQMSetupObjects(), EcalBarrel, double(timestamp_.iLumi), double(scal_pu));
    }
  }

  template <typename DigiCollection>
  void OccupancyTask::runOnDigis(DigiCollection const& _digis, Collections _collection) {
    MESet& meDigi(MEs_.at("Digi"));
    MESet& meDigiProjEta(MEs_.at("DigiProjEta"));
    MESet& meDigiProjPhi(MEs_.at("DigiProjPhi"));
    MESet& meDigiAll(MEs_.at("DigiAll"));
    MESet& meDigiAllByLumi(MEs_.at("DigiAllByLumi"));
    MESet& meDigiDCC(MEs_.at("DigiDCC"));
    MESet& meDigi1D(MEs_.at("Digi1D"));
    MESet& meTrendNDigi(MEs_.at("TrendNDigi"));
    MESet* meAELoss = nullptr;
    MESet* meAEReco = nullptr;
    if (onlineMode_) {
      meAELoss = &MEs_.at("AELoss");
      meAEReco = &MEs_.at("AEReco");
    }
    std::for_each(_digis.begin(), _digis.end(), [&](typename DigiCollection::Digi const& digi) {
      DetId id(digi.id());
      meDigi.fill(getEcalDQMSetupObjects(), id);
      meDigiProjEta.fill(getEcalDQMSetupObjects(), id);
      meDigiProjPhi.fill(getEcalDQMSetupObjects(), id);
      meDigiAll.fill(getEcalDQMSetupObjects(), id);
      meDigiAllByLumi.fill(getEcalDQMSetupObjects(), id);
      meDigiDCC.fill(getEcalDQMSetupObjects(), id);
      if (onlineMode_) {
        meAELoss->fill(getEcalDQMSetupObjects(), id);
        meAEReco->fill(getEcalDQMSetupObjects(), id);
      }
    });

    int iSubdet(_collection == kEBDigi ? EcalBarrel : EcalEndcap);
    meDigi1D.fill(getEcalDQMSetupObjects(), iSubdet, double(_digis.size()));
    meTrendNDigi.fill(getEcalDQMSetupObjects(), iSubdet, double(timestamp_.iLumi), double(_digis.size()));
  }

  void OccupancyTask::runOnTPDigis(EcalTrigPrimDigiCollection const& _digis) {
    //    MESet& meTPDigiAll(MEs_.at("TPDigiAll"));
    //    MESet& meTPDigiProjEta(MEs_.at("TPDigiProjEta"));
    //    MESet& meTPDigiProjPhi(MEs_.at("TPDigiProjPhi"));
    MESet& meTPDigiRCT(MEs_.at("TPDigiRCT"));
    MESet& meTPDigiThrAll(MEs_.at("TPDigiThrAll"));
    MESet& meTPDigiThrAllByLumi(MEs_.at("TPDigiThrAllByLumi"));
    MESet& meTPDigiThrProjEta(MEs_.at("TPDigiThrProjEta"));
    MESet& meTPDigiThrProjPhi(MEs_.at("TPDigiThrProjPhi"));
    MESet& meTrendNTPDigi(MEs_.at("TrendNTPDigi"));

    double nFilteredEB(0.);
    double nFilteredEE(0.);

    std::for_each(_digis.begin(), _digis.end(), [&](EcalTrigPrimDigiCollection::value_type const& digi) {
      EcalTrigTowerDetId const& id(digi.id());
      //       meTPDigiProjEta.fill(id);
      //       meTPDigiProjPhi.fill(id);
      //       meTPDigiAll.fill(id);
      if (digi.compressedEt() > tpThreshold_) {
        meTPDigiThrProjEta.fill(getEcalDQMSetupObjects(), id);
        meTPDigiThrProjPhi.fill(getEcalDQMSetupObjects(), id);
        meTPDigiThrAll.fill(getEcalDQMSetupObjects(), id);
        meTPDigiThrAllByLumi.fill(getEcalDQMSetupObjects(), id);
        meTPDigiRCT.fill(getEcalDQMSetupObjects(), id);
        if (id.subDet() == EcalBarrel)
          nFilteredEB += 1.;
        else
          nFilteredEE += 1.;
      }
    });

    meTrendNTPDigi.fill(getEcalDQMSetupObjects(), EcalBarrel, double(timestamp_.iLumi), nFilteredEB);
    meTrendNTPDigi.fill(getEcalDQMSetupObjects(), EcalEndcap, double(timestamp_.iLumi), nFilteredEE);
  }

  void OccupancyTask::runOnRecHits(EcalRecHitCollection const& _hits, Collections _collection) {
    MESet& meRecHitAll(MEs_.at("RecHitAll"));
    MESet& meRecHitProjEta(MEs_.at("RecHitProjEta"));
    MESet& meRecHitProjPhi(MEs_.at("RecHitProjPhi"));
    MESet& meRecHitThrAll(MEs_.at("RecHitThrAll"));
    MESet& meRecHitThrAllByLumi(MEs_.at("RecHitThrAllByLumi"));
    MESet& meRecHitThrmvp(MEs_.at("RecHitThrmvp"));
    MESet& meRecHitThrpm(MEs_.at("RecHitThrpm"));
    MESet& meRecHitThrProjEta(MEs_.at("RecHitThrProjEta"));
    MESet& meRecHitThrProjPhi(MEs_.at("RecHitThrProjPhi"));
    MESet& meRecHitThr1D(MEs_.at("RecHitThr1D"));
    MESet& meTrendNRecHitThr(MEs_.at("TrendNRecHitThr"));

    uint32_t mask(~(0x1 << EcalRecHit::kGood));
    double nFiltered(0.);

    float nRHThrp(0), nRHThrm(0);
    int iSubdet(_collection == kEBRecHit ? EcalBarrel : EcalEndcap);
    std::for_each(_hits.begin(), _hits.end(), [&](EcalRecHitCollection::value_type const& hit) {
      DetId id(hit.id());

      meRecHitAll.fill(getEcalDQMSetupObjects(), id);
      meRecHitProjEta.fill(getEcalDQMSetupObjects(), id);
      meRecHitProjPhi.fill(getEcalDQMSetupObjects(), id);

      if (!hit.checkFlagMask(mask) && hit.energy() > recHitThreshold_) {
        meRecHitThrProjEta.fill(getEcalDQMSetupObjects(), id);
        meRecHitThrProjPhi.fill(getEcalDQMSetupObjects(), id);
        meRecHitThrAll.fill(getEcalDQMSetupObjects(), id);
        meRecHitThrAllByLumi.fill(getEcalDQMSetupObjects(), id);
        nFiltered += 1.;
        bool isPlusFar(iSubdet == EcalBarrel ? (EBDetId(id).iphi() > 100 && EBDetId(id).iphi() < 280) : zside(id) > 0);
        if (isPlusFar)
          nRHThrp++;
        else
          nRHThrm++;
      }
    });

    meRecHitThr1D.fill(getEcalDQMSetupObjects(), iSubdet, nFiltered);
    meTrendNRecHitThr.fill(getEcalDQMSetupObjects(), iSubdet, double(timestamp_.iLumi), nFiltered);
    meRecHitThrmvp.fill(getEcalDQMSetupObjects(), iSubdet, nRHThrp, nRHThrm);
    meRecHitThrpm.fill(getEcalDQMSetupObjects(), iSubdet, nRHThrp - nRHThrm);
  }

  DEFINE_ECALDQM_WORKER(OccupancyTask);
}  // namespace ecaldqm
