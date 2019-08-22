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
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include <cmath>

namespace spr {

  // All types of Hit Collections
  template <typename T>
  std::vector<typename T::const_iterator> findHit(edm::Handle<T>& hits, DetId thisDet, bool debug = false);

  std::vector<std::vector<PCaloHit>::const_iterator> findHit(std::vector<PCaloHit>& hits,
                                                             DetId thisDet,
                                                             bool debug = false);

  template <typename T>
  void findHit(edm::Handle<T>& hits, DetId thisDet, std::vector<typename T::const_iterator>& hit, bool debug = false);

  // For EB and EE RecHit Collection
  void find(edm::Handle<EcalRecHitCollection>& hits,
            DetId thisDet,
            std::vector<EcalRecHitCollection::const_iterator>& hit,
            bool debug = false);

  // For Hcal RecHit Collection
  void find(edm::Handle<HBHERecHitCollection>& hits,
            DetId thisDet,
            std::vector<HBHERecHitCollection::const_iterator>& hit,
            bool debug = false);

  // For simHit Collection
  void find(edm::Handle<edm::PCaloHitContainer>& hits,
            DetId thisDet,
            std::vector<edm::PCaloHitContainer::const_iterator>& hit,
            bool debug = false);
}  // namespace spr

#include "Calibration/IsolatedParticles/interface/FindCaloHit.icc"

#endif
