#ifndef BTauReco_JetTracksAssociation_h
#define BTauReco_JetTracksAssociation_h
// \class JetTracksAssociation
// 
// \short association of tracks to jet (was JetWithTracks)
// 
//

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <vector>

namespace reco {
  typedef
  std::vector<std::pair<edm::RefToBase<Jet>,reco::TrackRefVector> > 
     JetTracksAssociationCollection;
  
  typedef
  JetTracksAssociationCollection::value_type JetTracksAssociation;
  
  typedef
  edm::Ref<JetTracksAssociationCollection> JetTracksAssociationRef;
  
  typedef
  edm::RefProd<JetTracksAssociationCollection> JetTracksAssociationRefProd;
  
  typedef
  edm::RefVector<JetTracksAssociationCollection> JetTracksAssociationRefVector; 
}
#endif
