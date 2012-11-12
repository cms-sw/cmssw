#include "../interface/OccupancyTask.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

namespace ecaldqm {

  OccupancyTask::OccupancyTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "OccupancyTask"),
    recHitThreshold_(_workerParams.getUntrackedParameter<double>("recHitThreshold")),
    tpThreshold_(_workerParams.getUntrackedParameter<double>("tpThreshold"))
  {
    collectionMask_[kEBDigi] = true;
    collectionMask_[kEEDigi] = true;
    collectionMask_[kTrigPrimDigi] = true;
    collectionMask_[kEBRecHit] = true;
    collectionMask_[kEERecHit] = true;
  }

  OccupancyTask::~OccupancyTask()
  {
  }

  bool
  OccupancyTask::filterRunType(const std::vector<short>& _runType)
  {
    for(int iFED(0); iFED < 54; iFED++){
      if(_runType[iFED] == EcalDCCHeaderBlock::COSMIC ||
	 _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
	 _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
	 _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
	 _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
	 _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL) return true;
    }

    return false;
  }

  void
  OccupancyTask::runOnDigis(const EcalDigiCollection &_digis, Collections _collection)
  {
    MESet* meDigi(MEs_["Digi"]);
    MESet* meDigiProjEta(MEs_["DigiProjEta"]);
    MESet* meDigiProjPhi(MEs_["DigiProjPhi"]);
    MESet* meDigiAll(MEs_["DigiAll"]);
    MESet* meDigiDCC(MEs_["DigiDCC"]);
    MESet* meDigi1D(MEs_["Digi1D"]);
    MESet* meTrendNDigi(online ? MEs_["TrendNDigi"] : 0);

    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      DetId id(digiItr->id());
      meDigi->fill(id);
      meDigiProjEta->fill(id);
      meDigiProjPhi->fill(id);
      meDigiAll->fill(id);
      meDigiDCC->fill(id);
    }

    unsigned iSubdet(_collection == kEBDigi ? BinService::kEB + 1 : BinService::kEE + 1);
    meDigi1D->fill(iSubdet, double(_digis.size()));
    if(online) meTrendNDigi->fill(iSubdet, double(iLumi), double(_digis.size()));
  }

  void
  OccupancyTask::runOnTPDigis(const EcalTrigPrimDigiCollection &_digis)
  {
    //    MESet* meTPDigiAll(MEs_["TPDigiAll"]);
    //    MESet* meTPDigiProjEta(MEs_["TPDigiProjEta"]);
    //    MESet* meTPDigiProjPhi(MEs_["TPDigiProjPhi"]);
    MESet* meTPDigiThrAll(MEs_["TPDigiThrAll"]);
    MESet* meTPDigiThrProjEta(MEs_["TPDigiThrProjEta"]);
    MESet* meTPDigiThrProjPhi(MEs_["TPDigiThrProjPhi"]);
    MESet* meTrendNTPDigi(online ? MEs_["TrendNTPDigi"] : 0);

    double nFilteredEB(0.);
    double nFilteredEE(0.);

    for(EcalTrigPrimDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      EcalTrigTowerDetId const& id(digiItr->id());

//       meTPDigiProjEta->fill(id);
//       meTPDigiProjPhi->fill(id);
//       meTPDigiAll->fill(id);

      if(digiItr->compressedEt() > tpThreshold_){
	meTPDigiThrProjEta->fill(id);
	meTPDigiThrProjPhi->fill(id);
	meTPDigiThrAll->fill(id);
        if(id.subDet() == EcalBarrel) nFilteredEB += 1.;
        else nFilteredEE += 1.;
      }
    }

    if(online){
      meTrendNTPDigi->fill(unsigned(BinService::kEB + 1), double(iLumi), nFilteredEB);
      meTrendNTPDigi->fill(unsigned(BinService::kEE + 1), double(iLumi), nFilteredEE);
    }
  }

  void
  OccupancyTask::runOnRecHits(const EcalRecHitCollection &_hits, Collections _collection)
  {
    MESet* meRecHitAll(MEs_["RecHitAll"]);
    MESet* meRecHitProjEta(MEs_["RecHitProjEta"]);
    MESet* meRecHitProjPhi(MEs_["RecHitProjPhi"]);
    MESet* meRecHitThrAll(MEs_["RecHitThrAll"]);
    MESet* meRecHitThrProjEta(MEs_["RecHitThrProjEta"]);
    MESet* meRecHitThrProjPhi(MEs_["RecHitThrProjPhi"]);
    MESet* meRecHitThr1D(MEs_["RecHitThr1D"]);
    MESet* meTrendNRecHitThr(online ? MEs_["TrendNRecHitThr"] : 0);

    uint32_t mask(~(0x1 << EcalRecHit::kGood));
    double nFiltered(0.);

    for(EcalRecHitCollection::const_iterator hitItr(_hits.begin()); hitItr != _hits.end(); ++hitItr){
      DetId id(hitItr->id());

      meRecHitAll->fill(id);
      meRecHitProjEta->fill(id);
      meRecHitProjPhi->fill(id);

      if(!hitItr->checkFlagMask(mask) && hitItr->energy() > recHitThreshold_){
	meRecHitThrProjEta->fill(id);
	meRecHitThrProjPhi->fill(id);
	meRecHitThrAll->fill(id);
        nFiltered += 1.;
      }
    }

    unsigned iSubdet(_collection == kEBRecHit ? BinService::kEB + 1 : BinService::kEE + 1);
    meRecHitThr1D->fill(iSubdet, nFiltered);
    if(online) meTrendNRecHitThr->fill(iSubdet, double(iLumi), nFiltered);
  }

  DEFINE_ECALDQM_WORKER(OccupancyTask);
}

