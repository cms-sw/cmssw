// F.R.
// $Id: CaloJetfwd.h,v 1.4 2006/05/24 00:40:43 fedor Exp $
#ifndef JetReco_CaloJetfwd_h
#define JetReco_CaloJetfwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

class CaloJet;
/// collection of CaloJet objects 
typedef std::vector<CaloJet> CaloJetCollection;
/// edm references
typedef edm::Ref<CaloJetCollection> CaloJetRef;
typedef edm::RefVector<CaloJetCollection> CaloJetRefVector;
typedef edm::RefProd<CaloJetCollection> CaloJetRefProd;
#endif
