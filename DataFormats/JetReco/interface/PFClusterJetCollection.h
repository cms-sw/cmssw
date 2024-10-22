#ifndef DataFormats_JetReco_PFClusterJetCollection_h
#define DataFormats_JetReco_PFClusterJetCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/JetReco/interface/PFClusterJet.h"

namespace reco {

  /// collection of PFClusterJet objects
  typedef std::vector<PFClusterJet> PFClusterJetCollection;
  /// edm references
  typedef edm::Ref<PFClusterJetCollection> PFClusterJetRef;
  typedef edm::FwdRef<PFClusterJetCollection> PFClusterJetFwdRef;
  typedef edm::FwdPtr<PFClusterJet> PFClusterJetFwdPtr;
  typedef edm::RefVector<PFClusterJetCollection> PFClusterJetRefVector;
  typedef std::vector<edm::FwdRef<PFClusterJetCollection> > PFClusterJetFwdRefVector;
  typedef std::vector<edm::FwdPtr<PFClusterJet> > PFClusterJetFwdPtrVector;
  typedef edm::RefProd<PFClusterJetCollection> PFClusterJetRefProd;

}  // namespace reco

#endif
