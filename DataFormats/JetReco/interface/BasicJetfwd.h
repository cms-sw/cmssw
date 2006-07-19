// F.R.
// $Id: BasicJetfwd.h,v 1.6 2006/06/27 23:15:06 fedor Exp $
#ifndef JetReco_BasicJetfwd_h
#define JetReco_BasicJetfwd_h
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class BasicJet;
  /// collection of BasicJet objects 
  typedef std::vector<BasicJet> BasicJetCollection;
  /// edm references
  typedef edm::Ref<BasicJetCollection> BasicJetRef;
  typedef edm::RefVector<BasicJetCollection> BasicJetRefVector;
  typedef edm::RefProd<BasicJetCollection> BasicJetRefProd;
}
#endif
