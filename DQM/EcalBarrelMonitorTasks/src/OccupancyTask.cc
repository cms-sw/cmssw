#include "../interface/OccupancyTask.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

namespace ecaldqm {

  OccupancyTask::OccupancyTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "OccupancyTask"),
    recHitThreshold_(_workerParams.getUntrackedParameter<double>("recHitThreshold")),
    tpThreshold_(_workerParams.getUntrackedParameter<double>("tpThreshold"))
  {
    collectionMask_ =
      (0x1 << kEBDigi) |
      (0x1 << kEEDigi) |
      (0x1 << kTrigPrimDigi) |
      (0x1 << kEBRecHit) |
      (0x1 << kEERecHit);
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
  OccupancyTask::runOnDigis(const EcalDigiCollection &_digis)
  {
    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      DetId id(digiItr->id());
      MEs_[kDigi]->fill(id);
      MEs_[kDigiProjEta]->fill(id);
      MEs_[kDigiProjPhi]->fill(id);
      MEs_[kDigiAll]->fill(id);
      MEs_[kDigiDCC]->fill(id);
    }
  }

  void
  OccupancyTask::runOnTPDigis(const EcalTrigPrimDigiCollection &_digis)
  {
    for(EcalTrigPrimDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){
      EcalTrigTowerDetId const& id(digiItr->id());

      MEs_[kTPDigi]->fill(id);
      MEs_[kTPDigiProjEta]->fill(id);
      MEs_[kTPDigiProjPhi]->fill(id);

      if(digiItr->compressedEt() > tpThreshold_){
	MEs_[kTPDigiThr]->fill(id);
	MEs_[kTPDigiThrProjEta]->fill(id);
	MEs_[kTPDigiThrProjPhi]->fill(id);
	MEs_[kTPDigiThrAll]->fill(id);
      }
    }
  }

  void
  OccupancyTask::runOnRecHits(const EcalRecHitCollection &_hits, Collections _collection)
  {
    uint32_t mask(~(0x1 << EcalRecHit::kGood));

    for(EcalRecHitCollection::const_iterator hitItr(_hits.begin()); hitItr != _hits.end(); ++hitItr){
      DetId id(hitItr->id());

//       MEs_[kRecHit]->fill(id);
//       MEs_[kRecHitProjEta]->fill(id);
//       MEs_[kRecHitProjPhi]->fill(id);

      if(!hitItr->checkFlagMask(mask) && hitItr->energy() > recHitThreshold_){
	MEs_[kRecHitThr]->fill(id);
	MEs_[kRecHitThrProjEta]->fill(id);
	MEs_[kRecHitThrProjPhi]->fill(id);
	MEs_[kRecHitThrAll]->fill(id);
      }
    }

    if(_collection == kEBRecHit)
      MEs_[kRecHit1D]->fill(unsigned(BinService::kEB) + 1, float(_hits.size()));
    else
      MEs_[kRecHit1D]->fill(unsigned(BinService::kEE) + 1, float(_hits.size()));
  }

  /*static*/
  void
  OccupancyTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["Digi"] = kDigi;
    _nameToIndex["DigiProjEta"] = kDigiProjEta;
    _nameToIndex["DigiProjPhi"] = kDigiProjPhi;
    _nameToIndex["DigiAll"] = kDigiAll;
    _nameToIndex["DigiDCC"] = kDigiDCC;
    _nameToIndex["RecHit1D"] = kRecHit1D;
    _nameToIndex["RecHitThr"] = kRecHitThr;
    _nameToIndex["RecHitThrProjEta"] = kRecHitThrProjEta;
    _nameToIndex["RecHitThrProjPhi"] = kRecHitThrProjPhi;
    _nameToIndex["RecHitThrAll"] = kRecHitThrAll;
    _nameToIndex["TPDigi"] = kTPDigi;
    _nameToIndex["TPDigiProjEta"] = kTPDigiProjEta;
    _nameToIndex["TPDigiProjPhi"] = kTPDigiProjPhi;
    _nameToIndex["TPDigiThr"] = kTPDigiThr;
    _nameToIndex["TPDigiThrProjEta"] = kTPDigiThrProjEta;
    _nameToIndex["TPDigiThrProjPhi"] = kTPDigiThrProjPhi;
    _nameToIndex["TPDigiThrAll"] = kTPDigiThrAll;
  }

  DEFINE_ECALDQM_WORKER(OccupancyTask);
}

