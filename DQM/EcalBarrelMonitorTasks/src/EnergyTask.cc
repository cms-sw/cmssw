#include "../interface/EnergyTask.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

namespace ecaldqm {

  EnergyTask::EnergyTask(const edm::ParameterSet &_params, const edm::ParameterSet& _paths) :
    DQWorkerTask(_params, _paths, "EnergyTask"),
    topology_(0),
    isPhysicsRun_(false),
    threshS9_(0.)
  {
    collectionMask_ = 
      (0x1 << kRun) |
      (0x1 << kEBRecHit) |
      (0x1 << kEERecHit);

    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));

    isPhysicsRun_ = taskParams.getUntrackedParameter<bool>("isPhysicsRun");
    threshS9_ = taskParams.getUntrackedParameter<double>("threshS9");

    std::map<std::string, std::string> replacements;
    if(!isPhysicsRun_) replacements["oot"] = " (outOfTime)";
    else replacements["oot"] = "";

    for(unsigned iME(0); iME < nMESets; iME++)
      MEs_[iME]->name(replacements);
  }

  EnergyTask::~EnergyTask()
  {
  }

  void
  EnergyTask::beginRun(const edm::Run &, const edm::EventSetup &_es)
  {
    edm::ESHandle<CaloTopology> topoHndl;
    _es.get<CaloTopologyRecord>().get(topoHndl);
    topology_ = topoHndl.product();
    if(!topology_)
      throw cms::Exception("EventSetup") << "CaloTopology missing" << std::endl;
  }
  
  bool
  EnergyTask::filterRunType(const std::vector<short>& _runType)
  {
    for(int iFED(0); iFED < 54; iFED++){
      if ( _runType[iFED] == EcalDCCHeaderBlock::COSMIC ||
           _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
           _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
           _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
           _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
           _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL ) return true;
    }

    return false;
  }

  void 
  EnergyTask::runOnRecHits(const EcalRecHitCollection &_hits)
  {
    uint32_t mask(~(0x1 << EcalRecHit::kGood));

    for(EcalRecHitCollection::const_iterator hitItr(_hits.begin()); hitItr != _hits.end(); ++hitItr){

      if(isPhysicsRun_ && hitItr->checkFlagMask(mask)) continue;

      float energy(isPhysicsRun_ ? hitItr->energy() : hitItr->outOfTimeEnergy());

      if ( energy < 0. ) energy = 0.0;

      DetId id(hitItr->id());

      MEs_[kHitMap]->fill(id, energy);
      MEs_[kHitMapAll]->fill(id, energy);
      MEs_[kHit]->fill(id, energy);
      MEs_[kHitAll]->fill(id, energy);

      // look for the seeds
      float e3x3(energy);
      bool isSeed = true;

      EcalRecHitCollection::const_iterator neighborItr;
      float neighborE;
      std::vector<DetId> window(topology_->getWindow(id, 3, 3));
      for(std::vector<DetId>::iterator idItr(window.begin()); idItr != window.end(); ++idItr){
	if((neighborItr = _hits.find(*idItr)) == _hits.end()) continue;
	if(isPhysicsRun_ && neighborItr->checkFlagMask(mask)) continue;
	neighborE = isPhysicsRun_ ? neighborItr->energy() : neighborItr->outOfTimeEnergy();
	if(neighborE > energy){
	  isSeed = false;
	  break;
	}
	e3x3 += neighborE;
      }

      if(!isSeed) continue;

      if ( e3x3 >= threshS9_ )
	MEs_[kMiniCluster]->fill(id, e3x3);

    }
  }

  /*static*/
  void
  EnergyTask::setMEData(std::vector<MEData>& _data)
  {
    BinService::AxisSpecs xaxis, zaxis;

    zaxis.low = 0.;
    zaxis.high = 10.;
    _data[kHitMap] = MEData("HitMap", BinService::kSM, BinService::kCrystal, MonitorElement::DQM_KIND_TPROFILE2D, 0, 0, &zaxis);
    _data[kHitMapAll] = MEData("HitMap", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TPROFILE2D, 0, 0, &zaxis);

    xaxis.nbins = 50;
    xaxis.low = 0.;
    xaxis.high = 20.;
    _data[kHit] = MEData("Hit", BinService::kSM, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);
    _data[kHitAll] = MEData("Hit", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);

    _data[kMiniCluster] = MEData("MiniCluster", BinService::kSM, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);
  }

  DEFINE_ECALDQM_WORKER(EnergyTask);
}


