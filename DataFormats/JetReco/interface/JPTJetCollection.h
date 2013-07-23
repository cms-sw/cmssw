// F.R.
// $Id: JPTJetCollection.h,v 1.1 2010/03/04 13:16:25 kodolova Exp $
#ifndef JetReco_JPTJetCollection_h
#define JetReco_JPTJetCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/JPTJet.h"

namespace reco {
  /// collection of CaloJet objects 
  typedef std::vector<JPTJet> JPTJetCollection;
  /// edm references
  typedef edm::Ref<JPTJetCollection> JPTJetRef;
  typedef edm::RefVector<JPTJetCollection> JPTJetRefVector;
  typedef edm::RefProd<JPTJetCollection> JPTJetRefProd;
}
#endif
