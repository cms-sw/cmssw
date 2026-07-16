#ifndef EgammaReco_ElectronFwd_h
#define EgammaReco_ElectronFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  namespace io_v1 {
    class Electron;
  }
  using Electron = io_v1::Electron;

  /// collectin of Electron objects
  typedef std::vector<Electron> ElectronCollection;

  /// reference to an object in a collection of Electron objects
  typedef edm::Ref<ElectronCollection> ElectronRef;

  /// reference to a collection of Electron objects
  typedef edm::RefProd<ElectronCollection> ElectronRefProd;

  /// vector of objects in the same collection of Electron objects
  typedef edm::RefVector<ElectronCollection> ElectronRefVector;

  /// iterator over a vector of reference to Electron objects
  typedef ElectronRefVector::iterator electron_iterator;
}  // namespace reco

#endif
