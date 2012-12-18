#include "../interface/OccupancyTask.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

namespace ecaldqm {

  OccupancyTask::OccupancyTask(const edm::ParameterSet &_params, const edm::ParameterSet& _paths) :
    DQWorkerTask(_params, _paths, "OccupancyTask"),
    recHitThreshold_(0.),
    tpThreshold_(0.)
  {
    collectionMask_ =
      (0x1 << kEBDigi) |
      (0x1 << kEEDigi) |
      (0x1 << kTrigPrimDigi) |
      (0x1 << kEBRecHit) |
      (0x1 << kEERecHit);

    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));

    recHitThreshold_ = taskParams.getUntrackedParameter<double>("recHitThreshold");
    tpThreshold_ = taskParams.getUntrackedParameter<double>("tpThreshold", 1.0);
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
      MEs_[kRecHit1D]->fill((unsigned)BinService::kEB + 1, float(_hits.size()));
    else
      MEs_[kRecHit1D]->fill((unsigned)BinService::kEE + 1, float(_hits.size()));
  }

  /*static*/
  void
  OccupancyTask::setMEData(std::vector<MEData>& _data)
  {
    BinService::AxisSpecs axis;

    _data[kDigi] = MEData("Digi", BinService::kSM, BinService::kCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kDigiProjEta] = MEData("Digi", BinService::kEcal3P, BinService::kProjEta, MonitorElement::DQM_KIND_TH1F);
    _data[kDigiProjPhi] = MEData("Digi", BinService::kEcal3P, BinService::kProjPhi, MonitorElement::DQM_KIND_TH1F);
    _data[kDigiAll] = MEData("Digi", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kDigiDCC] = MEData("DigiDCC", BinService::kEcal2P, BinService::kDCC, MonitorElement::DQM_KIND_TH1F);

    axis.nbins = 100;
    axis.low = 0.;
    axis.high = 6000.;
    _data[kRecHit1D] = MEData("RecHit1D", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
    _data[kRecHitThr] = MEData("RecHitThr", BinService::kSM, BinService::kCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kRecHitThrProjEta] = MEData("RecHitThr", BinService::kEcal3P, BinService::kProjEta, MonitorElement::DQM_KIND_TH1F);
    _data[kRecHitThrProjPhi] = MEData("RecHitThr", BinService::kEcal3P, BinService::kProjPhi, MonitorElement::DQM_KIND_TH1F);
    _data[kRecHitThrAll] = MEData("RecHitThr", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kTPDigi] = MEData("TPDigi", BinService::kSM, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH2F);
    _data[kTPDigiProjEta] = MEData("TPDigi", BinService::kEcal3P, BinService::kProjEta, MonitorElement::DQM_KIND_TH1F);
    _data[kTPDigiProjPhi] = MEData("TPDigi", BinService::kEcal3P, BinService::kProjPhi, MonitorElement::DQM_KIND_TH1F);
    _data[kTPDigiThr] = MEData("TPDigiThr", BinService::kSM, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH2F);
    _data[kTPDigiThrProjEta] = MEData("TPDigiThr", BinService::kEcal3P, BinService::kProjEta, MonitorElement::DQM_KIND_TH1F);
    _data[kTPDigiThrProjPhi] = MEData("TPDigiThr", BinService::kEcal3P, BinService::kProjPhi, MonitorElement::DQM_KIND_TH1F);
    _data[kTPDigiThrAll] = MEData("TPDigiThr", BinService::kEcal3P, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH2F);
  }

  DEFINE_ECALDQM_WORKER(OccupancyTask);
}

