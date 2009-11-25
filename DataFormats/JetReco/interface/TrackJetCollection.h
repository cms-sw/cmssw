#ifndef DataFormats_JetReco_TrackJetCollection_h
#define DataFormats_JetReco_TrackJetCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/JetReco/interface/TrackJet.h"


namespace reco {

  /// collection of TrackJet objects 
  typedef std::vector<TrackJet> TrackJetCollection;
  /// edm references
  typedef edm::Ref<TrackJetCollection> TrackJetRef;
  typedef edm::RefVector<TrackJetCollection> TrackJetRefVector;
  typedef edm::RefProd<TrackJetCollection> TrackJetRefProd;

}

#endif
