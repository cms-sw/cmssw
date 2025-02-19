// F.R.
// $Id: PFJetCollection.h,v 1.5 2010/04/13 20:29:52 srappocc Exp $
#ifndef JetReco_PFJetCollection_h
#define JetReco_PFJetCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/PFJet.h" //INCLUDECHECKER:SKIP

namespace reco {
  class PFJet;
  /// collection of PFJet objects 
  typedef std::vector<PFJet> PFJetCollection;
  /// edm references
  typedef edm::Ref<PFJetCollection> PFJetRef;
  typedef edm::FwdRef<PFJetCollection> PFJetFwdRef;
  typedef edm::FwdPtr<PFJet> PFJetFwdPtr;
  typedef edm::RefVector<PFJetCollection> PFJetRefVector;
  typedef edm::RefProd<PFJetCollection> PFJetRefProd;
  typedef std::vector<edm::FwdRef<PFJetCollection> > PFJetFwdRefVector;
  typedef std::vector<edm::FwdPtr<PFJet> > PFJetFwdPtrVector;
}
#endif
