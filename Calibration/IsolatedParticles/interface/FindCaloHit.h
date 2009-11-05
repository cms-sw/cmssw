// -*- C++ -*
/* 
This function returns a vector of hits (RecHits/SimHits) in the hit collection
which have a DetId matching to a given DetId.

Original Author:  Seema Sharma
Created: August 2009
*/

#ifndef CalibrationIsolatedParticlesFindCaloHit_h
#define CalibrationIsolatedParticlesFindCaloHit_h

// system include files
#include <memory>
#include <cmath>
#include <string>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <cmath>

namespace spr {

  // All types of Hit Collections
  template <typename T>
  std::vector<typename T::const_iterator> findHit(edm::Handle<T>& hits, DetId thisDet, bool debug=false);

  // For EB and EE RecHit Collection
  std::vector<EcalRecHitCollection::const_iterator>   find(edm::Handle<EcalRecHitCollection>& hits,   DetId thisDet, bool debug=false);  

  // For Hcal RecHit Collection
  std::vector<HBHERecHitCollection::const_iterator>   find(edm::Handle<HBHERecHitCollection>& hits,   DetId thisDet, bool debug=false);

  // For simHit Collection
  std::vector<edm::PCaloHitContainer::const_iterator> find(edm::Handle<edm::PCaloHitContainer>& hits, DetId thisDet, bool debug=false);

}

#include "Calibration/IsolatedParticles/interface/FindCaloHit.icc"

#endif
