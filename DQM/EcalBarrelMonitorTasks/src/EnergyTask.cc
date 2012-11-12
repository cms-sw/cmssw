#include "../interface/EnergyTask.h"

#include "FWCore/Framework/interface/ESHandle.h"
//#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

namespace ecaldqm {

  EnergyTask::EnergyTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "EnergyTask"),
    //    topology_(0),
    isPhysicsRun_(_workerParams.getUntrackedParameter<bool>("isPhysicsRun"))/*,
    threshS9_(_workerParams.getUntrackedParameter<double>("threshS9"))*/
  {
    collectionMask_[kRun] = true;
    collectionMask_[kEBRecHit] = true;
    collectionMask_[kEERecHit] = true;
  }

  EnergyTask::~EnergyTask()
  {
  }

//   void
//   EnergyTask::beginRun(const edm::Run &, const edm::EventSetup &/*_es*/)
//   {
//     edm::ESHandle<CaloTopology> topoHndl;
//     _es.get<CaloTopologyRecord>().get(topoHndl);
//     topology_ = topoHndl.product();
//     if(!topology_)
//       throw cms::Exception("EventSetup") << "CaloTopology missing" << std::endl;
//   }
  
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
    MESet* meHitMap(MEs_["HitMap"]);
    MESet* meHitMapAll(MEs_["HitMapAll"]);
    MESet* meHit(MEs_["Hit"]);
    MESet* meHitAll(MEs_["HitAll"]);

    uint32_t notGood(~(0x1 << EcalRecHit::kGood));
    uint32_t neitherGoodNorOOT(~(0x1 << EcalRecHit::kGood |
                                 0x1 << EcalRecHit::kOutOfTime));

    for(EcalRecHitCollection::const_iterator hitItr(_hits.begin()); hitItr != _hits.end(); ++hitItr){

      if(isPhysicsRun_ && hitItr->checkFlagMask(notGood)) continue;
      if(!isPhysicsRun_ && hitItr->checkFlagMask(neitherGoodNorOOT)) continue;

      float energy(isPhysicsRun_ ? hitItr->energy() : hitItr->outOfTimeEnergy());

      if(energy < 0.) continue;

      DetId id(hitItr->id());

      meHitMap->fill(id, energy);
      meHitMapAll->fill(id, energy);
      meHit->fill(id, energy);
      meHitAll->fill(id, energy);

      // look for the seeds
//       float e3x3(energy);
//       bool isSeed = true;

//       EcalRecHitCollection::const_iterator neighborItr;
//       float neighborE;
//       std::vector<DetId> window(topology_->getWindow(id, 3, 3));
//       for(std::vector<DetId>::iterator idItr(window.begin()); idItr != window.end(); ++idItr){
// 	if((neighborItr = _hits.find(*idItr)) == _hits.end()) continue;
//         if(isPhysicsRun_ && neighborItr->checkFlagMask(notGood)) continue;
//         if(!isPhysicsRun_ && neighborItr->checkFlagMask(neitherGoodNorOOT)) continue;
// 	neighborE = isPhysicsRun_ ? neighborItr->energy() : neighborItr->outOfTimeEnergy();
// 	if(neighborE > energy){
// 	  isSeed = false;
// 	  break;
// 	}
// 	e3x3 += neighborE;
//       }

//       if(!isSeed) continue;

//       if ( e3x3 >= threshS9_ )
// 	MEs_[kMiniCluster]->fill(id, e3x3);

    }
  }

  DEFINE_ECALDQM_WORKER(EnergyTask);
}

