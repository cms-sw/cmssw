// F.R.
// $Id$
#ifndef JetReco_GenJetfwd_h
#define JetReco_GenJetfwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

class GenJet;
/// collection of GenJet objects 
typedef std::vector<GenJet> GenJetCollection;
/// edm references
typedef edm::Ref<GenJetCollection> GenJetRef;
typedef edm::RefVector<GenJetCollection> GenJetRefs;
typedef edm::RefProd<GenJetCollection> GenJetsRef;
#endif
