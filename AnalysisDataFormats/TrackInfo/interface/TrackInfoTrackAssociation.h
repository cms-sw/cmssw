#ifndef TrackInfoTrackAssociation_h
#define TrackInfoTrackAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco {
  
  // association map
  typedef edm::AssociationMap<edm::OneToOne<TrackCollection, TrackInfoCollection> > TrackInfoTrackAssociationCollection;
  
  typedef TrackInfoTrackAssociationCollection::value_type TrackInfoTrackAssociation;
  
  // reference to an object in a collection of SeedMap objects
  typedef edm::Ref<TrackInfoTrackAssociationCollection> TrackInfoTrackAssociationRef;
  
  // reference to a collection of SeedMap objects
  typedef edm::RefProd<TrackInfoTrackAssociationCollection> TrackInfoTrackAssociationRefProd;
      
  // vector of references to objects in the same colletion of SeedMap objects
  typedef edm::RefVector<TrackInfoTrackAssociationCollection> TrackInfoTrackAssociationRefVector;
	
}
 
#endif
