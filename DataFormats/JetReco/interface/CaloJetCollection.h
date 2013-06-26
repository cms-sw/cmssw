// F.R.
// $Id: CaloJetCollection.h,v 1.8 2010/04/13 20:29:52 srappocc Exp $
#ifndef JetReco_CaloJetCollection_h
#define JetReco_CaloJetCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/CaloJet.h"

namespace reco {
  /// collection of CaloJet objects 
  typedef std::vector<CaloJet> CaloJetCollection;
  /// edm references
  typedef edm::Ref<CaloJetCollection> CaloJetRef;
  typedef edm::FwdRef<CaloJetCollection> CaloJetFwdRef;
  typedef edm::FwdPtr<CaloJet> CaloJetFwdPtr;
  typedef edm::RefVector<CaloJetCollection> CaloJetRefVector;
  typedef edm::RefProd<CaloJetCollection> CaloJetRefProd;  
  typedef std::vector<edm::FwdRef<CaloJetCollection> > CaloJetFwdRefVector;
  typedef std::vector<edm::FwdPtr<CaloJet> > CaloJetFwdPtrVector;
}
#endif
