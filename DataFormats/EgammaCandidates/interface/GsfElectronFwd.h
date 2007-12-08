#ifndef EgammaReco_GsfElectronFwd_h
#define EgammaReco_GsfElectronFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

namespace reco {
  class GsfElectron;

  /// collection of GsfElectron objects
  typedef std::vector<GsfElectron> GsfElectronCollection;

  /// reference to an object in a collection of GsfElectron objects
  typedef edm::Ref<GsfElectronCollection> GsfElectronRef;

  /// reference to a collection of GsfElectron objects
  typedef edm::RefProd<GsfElectronCollection> GsfElectronRefProd;

  /// vector of objects in the same collection of GsfElectron objects
  typedef edm::RefVector<GsfElectronCollection> GsfElectronRefVector;

  /// iterator over a vector of reference to GsfElectron objects
  typedef GsfElectronRefVector::iterator pixelMatchGsfElectron_iterator;
}

#endif
