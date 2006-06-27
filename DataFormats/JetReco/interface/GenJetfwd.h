// F.R.
// $Id: GenJetfwd.h,v 1.4 2006/06/07 12:28:44 llista Exp $
#ifndef JetReco_GenJetfwd_h
#define JetReco_GenJetfwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class GenJet;
  /// collection of GenJet objects 
  typedef std::vector<GenJet> GenJetCollection;
  /// edm references
  typedef edm::Ref<GenJetCollection> GenJetRef;
  typedef edm::RefVector<GenJetCollection> GenJetRefVector;
  typedef edm::RefProd<GenJetCollection> GenJetRefProd;
}
#endif
