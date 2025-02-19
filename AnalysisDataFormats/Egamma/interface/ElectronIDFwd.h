#ifndef EgammaReco_ElectronIDFwd_h
#define EgammaReco_ElectronIDFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class ElectronID;

  /// collection of ElectronID objects
  typedef std::vector<ElectronID> ElectronIDCollection;

  /// reference to an object in a collection of ElectronID objects
  typedef edm::Ref<ElectronIDCollection> ElectronIDRef;

  /// reference to a collection of ElectronID objects
  typedef edm::RefProd<ElectronIDCollection> ElectronIDRefProd;

  /// vector of objects in the same collection of ElectronID objects
  typedef edm::RefVector<ElectronIDCollection> ElectronIDRefVector;

  /// iterator over a vector of reference to ElectronID objects
  typedef ElectronIDRefVector::iterator electronID_iterator;
}

#endif
