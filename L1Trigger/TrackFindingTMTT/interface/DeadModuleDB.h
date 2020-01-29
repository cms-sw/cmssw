#ifndef __DEADMODULEDB_H__
#define __DEADMODULEDB_H__

#include "DataFormats/Math/interface/deltaPhi.h"

#include <set>
#include <vector>
#include <map>
#include <utility>
#include <cmath>

using namespace std;

namespace TMTT {

class Settings;
class Stub;

//===============================================================================================================
/**
* This class allows you to declare parts of the Tracker to be dead.
* You should edit the C++ implementation of this class to specify which moduels are dead.
*
* Modify function defineDeadTrackRegions() to kill stubs in certain layers & angular regions of the tracker.
*
* Modify function defineDeadSectors() to reduce the number of layers cut on tracks in certain (eta,phi) sectors.
*/
//===============================================================================================================

class DeadModuleDB {

public:

  DeadModuleDB() {this->defineDeadTrackerRegions(); this->defineDeadSectors();}

  ~DeadModuleDB() {}

  // Does this stub lie in a dead region of the tracker?
  // This function is only invoked if cfg param DeadSimulate = True & used to kill stubs to emulate dead regions.
  bool killStub(const Stub* stub) const;

  // Should the required number of layers that tracks must have be reduced by 1 in the given sector, because of dead modules
  // within it? This function is only invoked if cfg param DeadReduceLayers = True.
  bool reduceLayerCut(unsigned int iPhiSec, unsigned int iEtaReg) const {return reduceLayerCut_.find( pair<unsigned int, unsigned int>(iPhiSec, iEtaReg) ) != reduceLayerCut_.end();} 

public: 

  // Define a class to describe regions of the tracker barrel that are dead.
  class DeadBarrelRegion {
  public:
    DeadBarrelRegion(float phiMin, float phiMax, float zMin, float zMax) : 
                     phiMin_(phiMin), phiMax_(phiMax), zMin_(zMin), zMax_(zMax) {}
  public:
    const float phiMin_;
    const float phiMax_;
    const float zMin_;
    const float zMax_;
  };

  // Define a class to describe regions of the tracker endcap that are dead.
  class DeadEndcapRegion {
  public:
    DeadEndcapRegion(float phiMin, float phiMax, float rMin, float rMax) : 
                     phiMin_(phiMin), phiMax_(phiMax), rMin_(rMin), rMax_(rMax) {}
  public:
    const float phiMin_;
    const float phiMax_;
    const float rMin_;
    const float rMax_;
  };

private:

  // Return the difference in azimuth of two angles (phi1 - phi2) in the range 0 to 2*PI.
  float deltaPhiPos(float phi1, float phi2) const {return (M_PI + reco::deltaPhi(phi1 - phi2 - M_PI, 0.) );}

  // Define regions of the tracker in which the modules are dead.
  // This function is only relevant if cfg param DeadSimulate = True, when any stubs in modules within these regions are killed,
  // so emulating MC with dead regions.
  void defineDeadTrackerRegions();

  // Define sectors in which the cut on number of layer numbers on tracks should be reduced by 1, because
  // they contain dead modules. This function is only relevant if cfg param DeadReduceLayers = True.
  // Ideally these sectors would be derived from the regions declared in definedDeadTrackerRegions().
  void defineDeadSectors();

private:

  // Regions of the tracker barrel & endcap which are dead.
  map< unsigned int, vector<DeadModuleDB::DeadBarrelRegion> >  deadBarrelRegions_;
  map< unsigned int, vector<DeadModuleDB::DeadEndcapRegion> >  deadEndcapRegions_;

  // Sectors in which layer cut should be reduced.
  set< pair<unsigned int, unsigned int> > reduceLayerCut_;

};

}
#endif

