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
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/OneToManyAssociation.h"
namespace reco {
  typedef
   edm::OneToManyAssociation<CaloJetCollection,
			     reco::TrackCollection> JetTracksAssociationCollection;
  
  typedef
  JetTracksAssociationCollection::value_type JetTracksAssociation;
  
  typedef
  edm::Ref<JetTracksAssociationCollection, JetTracksAssociation,
	   edm::refhelper::FindUsingAdvance<JetTracksAssociationCollection, 
					    JetTracksAssociation> > JetTracksAssociationRef;
  
  typedef
  edm::RefProd<JetTracksAssociationCollection> JetTracksAssociationRefProd;
  
  typedef
  edm::RefVector<JetTracksAssociationCollection, JetTracksAssociation,
		 edm::refhelper::FindUsingAdvance<JetTracksAssociationCollection, 
						  JetTracksAssociation> > JetTracksAssociationRefVector; 
}
#endif
