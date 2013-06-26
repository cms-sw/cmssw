#ifndef EgammaCandidates_SiStripElectronFwd_h
#define EgammaCandidates_SiStripElectronFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class SiStripElectron;

  /// collectin of SiStripElectron objects
  typedef std::vector<SiStripElectron> SiStripElectronCollection;

  /// reference to an object in a collection of SiStripElectron objects
  typedef edm::Ref<SiStripElectronCollection> SiStripElectronRef;

  /// reference to a collection of SiStripElectron objects
  typedef edm::RefProd<SiStripElectronCollection> SiStripElectronRefProd;

  /// vector of objects in the same collection of SiStripElectron objects
  typedef edm::RefVector<SiStripElectronCollection> SiStripElectronRefVector;

  /// iterator over a vector of reference to SiStripElectron objects
  typedef SiStripElectronRefVector::iterator siStripElectron_iterator;
}

#endif
