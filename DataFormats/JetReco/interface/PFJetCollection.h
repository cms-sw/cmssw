// F.R.
// $Id: PFJetfwd.h,v 1.1 2007/05/08 05:34:59 fedor Exp $
#ifndef JetReco_PFJetCollection_h
#define JetReco_PFJetCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/JetReco/interface/PFJet.h"

namespace reco {
  class PFJet;
  /// collection of PFJet objects 
  typedef std::vector<PFJet> PFJetCollection;
  /// edm references
  typedef edm::Ref<PFJetCollection> PFJetRef;
  typedef edm::RefVector<PFJetCollection> PFJetRefVector;
  typedef edm::RefProd<PFJetCollection> PFJetRefProd;
}
#endif
