// F.R.
// $Id$
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
typedef edm::RefVector<CaloJetCollection> CaloJetRefs;
typedef edm::RefProd<CaloJetCollection> CaloJetsRef;
#endif
