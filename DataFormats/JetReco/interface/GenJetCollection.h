// F.R.
#ifndef JetReco_GenJetCollection_h
#define JetReco_GenJetCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/GenJet.h"//INCLUDECHECKER:SKIP

namespace reco {
  class GenJet;
  /// collection of GenJet objects 
  typedef std::vector<GenJet> GenJetCollection;
  /// edm references
  typedef edm::Ref<GenJetCollection> GenJetRef;
  typedef edm::FwdRef<GenJetCollection> GenJetFwdRef;
  typedef edm::FwdPtr<GenJet> GenJetFwdPtr;
  typedef edm::RefVector<GenJetCollection> GenJetRefVector;
  typedef edm::RefProd<GenJetCollection> GenJetRefProd;
  typedef std::vector<edm::FwdRef<GenJetCollection> > GenJetFwdRefVector;
  typedef std::vector<edm::FwdPtr<GenJet> > GenJetFwdPtrVector;
}
#endif
