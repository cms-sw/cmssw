#include <L1Trigger/TrackFindingTMTT/interface/DeadModuleDB.h>
#include <L1Trigger/TrackFindingTMTT/interface/Stub.h>

//=== Does this stub lie in a dead region of the tracker?
//=== This function is only invoked if cfg param DeadSimulate = True & used to kill stubs to emulate dead regions.

namespace TMTT {

bool DeadModuleDB::killStub(const Stub* stub) const {
  bool dead = false;
  if (stub->barrel()) {
    if (deadBarrelRegions_.find(stub->layerId()) != deadBarrelRegions_.end()) {
      const vector<DeadModuleDB::DeadBarrelRegion>& bRegions = deadBarrelRegions_.at(stub->layerId());
      for (const DeadModuleDB::DeadBarrelRegion& bDead : bRegions) {
        if (this->deltaPhiPos(stub->phi(), bDead.phiMin_) < this->deltaPhiPos(bDead.phiMax_, bDead.phiMin_) &&
	    stub->z() > bDead.zMin_ && stub->z() < bDead.zMax_) dead = true;
      }
    }
  } else {
    if (deadEndcapRegions_.find(stub->layerId()) != deadEndcapRegions_.end()) {
      const vector<DeadModuleDB::DeadEndcapRegion>& eRegions = deadEndcapRegions_.at(stub->layerId());
      for (const DeadModuleDB::DeadEndcapRegion& eDead : eRegions) {
        if (this->deltaPhiPos(stub->phi(), eDead.phiMin_) < this->deltaPhiPos(eDead.phiMax_, eDead.phiMin_) &&
	    stub->r() > eDead.rMin_ && stub->r() < eDead.rMax_) dead = true;
      }
    }
  }
  return dead;
}

//=== Define regions of the tracker in which the modules are dead.
//=== This function is only involed if cfg param DeadSimulate = True, when any stubs in modules within these regions are killed,
//=== so emulating MC with dead regions.

void DeadModuleDB::defineDeadTrackerRegions() {
  // Example of how to declare a region of the tracker barrel as dead.
  const unsigned int iLayer = 5; // (1-6 = barrel layer; 11-15 = endcap A disk; 21-25 = endcap B disk)
//=== Should this stub be killed to emulate dead modules lying within a dead tracker region?
//=== This function is only invoked if cfg param DeadSimulate = True

  const float phiMin =    0.5;
  const float phiMax =    2.5;
  const float zMin   = -100.0;
  const float zMax =      0.0;
  deadBarrelRegions_[iLayer] .push_back( DeadModuleDB::DeadBarrelRegion(phiMin, phiMax, zMin, zMax) );
}

//=== Define sectors in which the cut on number of layer numbers on tracks should be reduced by 1, because
//=== they contain dead modules. This function is only invoked if cfg param DeadReduceLayers = True.
//=== Ideally these sectors would be derived from the regions declared in definedDeadTrackerRegions().

void DeadModuleDB::defineDeadSectors() {

  // Example of how to declare sectors as dead.
  // N.B. Histograms "AlgEffVsPhiSec", "AlgEffVsEtaSec", "AlgEffFitVsPhiSec*" and "AlgEffFitVsEtaSec*"
  // will help you determine which sectors you need to declare here. 
  for (unsigned int iPhiSec = 9; iPhiSec <= 13; iPhiSec++) { 
    for (unsigned int iEtaReg = 5; iEtaReg <= 8; iEtaReg++) {
      reduceLayerCut_.insert( pair<unsigned int, unsigned int> (iPhiSec, iEtaReg) );
    }
  }
}

}