// F.R.
#ifndef JetReco_GenericJetCollection_h
#define JetReco_GenericJetCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/GenericJet.h"  //INCLUDECHECKER:SKIP

namespace reco {
  class GenericJet;
  /// collection of GenericJet objects
  typedef std::vector<GenericJet> GenericJetCollection;
  /// edm references
  typedef edm::Ref<GenericJetCollection> GenericJetRef;
  typedef edm::FwdRef<GenericJetCollection> GenericJetFwdRef;
  typedef edm::RefVector<GenericJetCollection> GenericJetRefVector;
  typedef std::vector<edm::FwdRef<GenericJetCollection> > GenericJetFwdRefVector;
  typedef edm::RefProd<GenericJetCollection> GenericJetRefProd;
}  // namespace reco
#endif
