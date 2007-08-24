// F.R.
// $Id: GenericJetCollection.h,v 1.3 2007/08/20 17:53:30 fedor Exp $
#ifndef JetReco_GenericJetCollection_h
#define JetReco_GenericJetCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/GenericJet.h" //INCLUDECHECKER:SKIP

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
