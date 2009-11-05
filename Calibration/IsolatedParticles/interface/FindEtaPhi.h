// -*- C++ -*
/* 
This function returns a vector of hits (RecHits/SimHits) in the hit collection
which have a DetId matching to a given DetId.

Original Author:  Seema Sharma
Created: August 2009
*/

#ifndef CalibrationIsolatedParticlesFindEtaPhi_h
#define CalibrationIsolatedParticlesFindEtaPhi_h

// system include files
#include <memory>
#include <cmath>
#include <string>
#include <map>
#include <vector>

namespace spr {

  // For even number of NxN array
  struct EtaPhi {
    EtaPhi() {ntrys = 0;}
    int    ietaE[4], ietaW[4], iphiN[4], iphiS[4];
    int    ntrys;
  };
  
  EtaPhi getEtaPhi(int ieta, int iphi, bool debug=false);

}

#endif
