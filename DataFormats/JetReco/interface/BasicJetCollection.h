#ifndef JetReco_BasicJetCollection_h
#define JetReco_BasicJetCollection_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/JetReco/interface/BasicJet.h" //INCLUDECHECKER:SKIP

namespace reco {
  class BasicJet;
  /// collection of BasicJet objects 
  typedef std::vector<BasicJet> BasicJetCollection;
  /// edm references
  typedef edm::Ref<BasicJetCollection> BasicJetRef;
  typedef edm::RefVector<BasicJetCollection> BasicJetRefVector;
  typedef edm::RefProd<BasicJetCollection> BasicJetRefProd;
}
#endif
