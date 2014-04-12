#ifndef JetReco_JetCollection_h
#define JetReco_JetCollection_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/Jet.h"


namespace reco {
  /// edm references
  typedef edm::View<Jet> JetView;
  typedef edm::RefToBase<Jet> JetBaseRef;
  typedef edm::RefToBaseProd<reco::Jet> JetRefBaseProd;
}
#endif
