// F.R.
// $Id: GenJetCollection.h,v 1.3 2007/08/20 17:53:30 fedor Exp $
#ifndef JetReco_GenJetCollection_h
#define JetReco_GenJetCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/GenJet.h"//INCLUDECHECKER:SKIP

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
