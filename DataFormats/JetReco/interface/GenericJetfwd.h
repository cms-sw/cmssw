// F.R.
// $Id: GenJetfwd.h,v 1.5 2006/06/27 23:15:06 fedor Exp $
#ifndef JetReco_GenericJetfwd_h
#define JetReco_GenericJetfwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class GenericJet;
  /// collection of GenericJet objects 
  typedef std::vector<GenericJet> GenericJetCollection;
  /// edm references
  typedef edm::Ref<GenericJetCollection> GenericJetRef;
  typedef edm::RefVector<GenericJetCollection> GenericJetRefVector;
  typedef edm::RefProd<GenericJetCollection> GenericJetRefProd;
}
#endif
