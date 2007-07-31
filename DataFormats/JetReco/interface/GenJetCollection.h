// F.R.
// $Id: GenJetfwd.h,v 1.5 2006/06/27 23:15:06 fedor Exp $
#ifndef JetReco_GenJetCollection_h
#define JetReco_GenJetCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/JetReco/interface/GenJet.h"

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
