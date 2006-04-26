#ifndef BTauReco_JetTracksAssociation_h
#define BTauReco_JetTracksAssociation_h
// \class JetTracksAssociation
// 
// \short association of tracks to jet (was JetWithTracks)
// 
//

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/OneToManyAssociation.h"
namespace reco {
  typedef
      edm::OneToManyAssociation<CaloJetCollection,
      reco::TrackCollection> JetTracksAssociationCollection;

 typedef
      edm::OneToManyAssociation <CaloJetCollection,
      reco::TrackCollection>::value_type JetTracksAssociation;

   typedef
     edm::Ref<edm::OneToManyAssociation<CaloJetCollection,
       reco::TrackCollection> > JetTracksAssociationRef;

}
#endif
