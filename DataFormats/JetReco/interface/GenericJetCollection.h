// F.R.
// $Id: GenericJetCollection.h,v 1.1 2007/07/31 18:55:23 fedor Exp $
#ifndef JetReco_GenericJetCollection_h
#define JetReco_GenericJetCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/JetReco/interface/GenericJet.h"

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
