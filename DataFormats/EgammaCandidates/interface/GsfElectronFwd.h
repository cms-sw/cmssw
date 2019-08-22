
#ifndef EgammaReco_GsfElectronFwd_h
#define EgammaReco_GsfElectronFwd_h

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

namespace reco {

  class GsfElectron;

  /// collection of GsfElectron objects
  typedef std::vector<GsfElectron> GsfElectronCollection;
  //typedef GsfElectronCollection PixelMatchGsfElectronCollection ;

  /// reference to an object in a collection of GsfElectron objects
  typedef edm::Ref<GsfElectronCollection> GsfElectronRef;
  //typedef GsfElectronRef PixelMatchGsfElectronRef ;

  /// reference to a collection of GsfElectron objects
  typedef edm::RefProd<GsfElectronCollection> GsfElectronRefProd;
  //typedef GsfElectronRefProd PixelMatchGsfElectronRefProd ;

  /// vector of objects in the same collection of GsfElectron objects
  typedef edm::RefVector<GsfElectronCollection> GsfElectronRefVector;
  //typedef GsfElectronRefVector PixelMatchGsfElectronRefVector ;

  /// iterator over a vector of reference to GsfElectron objects
  typedef GsfElectronRefVector::iterator GsfElectron_iterator;
  //typedef GsfElectron_iterator PixelMatchGsfElectron_iterator ;

}  // namespace reco

#endif
