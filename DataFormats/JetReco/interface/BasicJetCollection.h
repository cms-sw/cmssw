#ifndef JetReco_BasicJetCollection_h
#define JetReco_BasicJetCollection_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/BasicJet.h"

namespace reco {
  /// collection of BasicJet objects
  typedef std::vector<BasicJet> BasicJetCollection;
  /// edm references
  typedef edm::Ref<BasicJetCollection> BasicJetRef;
  typedef edm::FwdRef<BasicJetCollection> BasicJetFwdRef;
  typedef edm::FwdPtr<BasicJet> BasicJetFwdPtr;
  typedef edm::RefVector<BasicJetCollection> BasicJetRefVector;
  typedef std::vector<edm::FwdRef<BasicJetCollection> > BasicJetFwdRefVector;
  typedef std::vector<edm::FwdPtr<BasicJet> > BasicJetFwdPtrVector;
  typedef edm::RefProd<BasicJetCollection> BasicJetRefProd;
}  // namespace reco
#endif
