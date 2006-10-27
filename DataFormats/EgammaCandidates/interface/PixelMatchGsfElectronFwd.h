#ifndef EgammaReco_PixelMatchGsfElectronFwd_h
#define EgammaReco_PixelMatchGsfElectronFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"

namespace reco {
  class PixelMatchGsfElectron;

  /// collection of PixelMatchGsfElectron objects
  typedef std::vector<PixelMatchGsfElectron> PixelMatchGsfElectronCollection;

  /// reference to an object in a collection of PixelMatchGsfElectron objects
  typedef edm::Ref<PixelMatchGsfElectronCollection> PixelMatchGsfElectronRef;

  /// reference to a collection of PixelMatchGsfElectron objects
  typedef edm::RefProd<PixelMatchGsfElectronCollection> PixelMatchGsfElectronRefProd;

  /// vector of objects in the same collection of PixelMatchGsfElectron objects
  typedef edm::RefVector<PixelMatchGsfElectronCollection> PixelMatchGsfElectronRefVector;

  /// iterator over a vector of reference to PixelMatchGsfElectron objects
  typedef PixelMatchGsfElectronRefVector::iterator pixelMatchGsfElectron_iterator;
}

#endif
