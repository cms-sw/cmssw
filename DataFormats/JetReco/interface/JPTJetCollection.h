// F.R.
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
