#ifndef EgammaReco_PixelMatchElectronFwd_h
#define EgammaReco_PixelMatchElectronFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchElectron.h"

namespace reco {
  class PixelMatchElectron;

  /// collection of PixelMatchElectron objects
  typedef std::vector<PixelMatchElectron> PixelMatchElectronCollection;

  /// reference to an object in a collection of PixelMatchElectron objects
  typedef edm::Ref<PixelMatchElectronCollection> PixelMatchElectronRef;

  /// reference to a collection of PixelMatchElectron objects
  typedef edm::RefProd<PixelMatchElectronCollection> PixelMatchElectronRefProd;

  /// vector of objects in the same collection of PixelMatchElectron objects
  typedef edm::RefVector<PixelMatchElectronCollection> PixelMatchElectronRefVector;

  /// iterator over a vector of reference to PixelMatchElectron objects
  typedef PixelMatchElectronRefVector::iterator pixelMatchElectron_iterator;
}

#endif
