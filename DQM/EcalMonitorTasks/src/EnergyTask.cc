#include "DQM/EcalMonitorTasks/interface/EnergyTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

namespace ecaldqm {
  EnergyTask::EnergyTask() : DQWorkerTask(), isPhysicsRun_(false) {}

  void EnergyTask::setParams(edm::ParameterSet const& _params) {
    isPhysicsRun_ = _params.getUntrackedParameter<bool>("isPhysicsRun");
  }

  bool EnergyTask::filterRunType(short const* _runType) {
    for (unsigned iFED(0); iFED != ecaldqm::nDCC; iFED++) {
      if (_runType[iFED] == EcalDCCHeaderBlock::COSMIC || _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
          _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
          _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL || _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
          _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL)
        return true;
    }

    return false;
  }

  void EnergyTask::beginEvent(edm::Event const& _evt, edm::EventSetup const& _es, bool const& ByLumiResetSwitch, bool&) {
    if (ByLumiResetSwitch) {
      MEs_.at("HitMapAllByLumi").reset(GetElectronicsMap());
    }
  }

  void EnergyTask::runOnRecHits(EcalRecHitCollection const& _hits) {
    MESet& meHitMap(MEs_.at("HitMap"));
    MESet& meHitMapAll(MEs_.at("HitMapAll"));
    MESet& meHitMapAllByLumi(MEs_.at("HitMapAllByLumi"));
    MESet& meHit(MEs_.at("Hit"));
    MESet& meHitAll(MEs_.at("HitAll"));

    uint32_t goodORPoorCalibBits(0x1 << EcalRecHit::kGood | 0x1 << EcalRecHit::kPoorCalib);
    uint32_t goodOROOTBits(0x1 << EcalRecHit::kGood | 0x1 << EcalRecHit::kOutOfTime);

    for (EcalRecHitCollection::const_iterator hitItr(_hits.begin()); hitItr != _hits.end(); ++hitItr) {
      if (isPhysicsRun_ && !hitItr->checkFlagMask(goodORPoorCalibBits))
        continue;
      if (!isPhysicsRun_ && !hitItr->checkFlagMask(goodOROOTBits))
        continue;

      float energy(hitItr->energy());

      if (energy < 0.)
        continue;

      DetId id(hitItr->id());

      meHitMap.fill(getEcalDQMSetupObjects(), id, energy);
      meHitMapAll.fill(getEcalDQMSetupObjects(), id, energy);
      meHitMapAllByLumi.fill(getEcalDQMSetupObjects(), id, energy);
      meHit.fill(getEcalDQMSetupObjects(), id, energy);
      meHitAll.fill(getEcalDQMSetupObjects(), id, energy);

      // look for the seeds
      //       float e3x3(energy);
      //       bool isSeed = true;

      //       EcalRecHitCollection::const_iterator neighborItr;
      //       float neighborE;
      //       std::vector<DetId> window(GetTopology()->getWindow(id, 3, 3));
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
}  // namespace ecaldqm
