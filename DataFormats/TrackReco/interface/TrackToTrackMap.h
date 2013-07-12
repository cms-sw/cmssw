#ifndef DataFormats_Track_TrackToTrackMap_H
#define DataFormats_Track_TrackToTrackMap_H

/** \class TrackToTrackMap
 *  No description available.
 *
 *  $Date: 2007/03/15 11:12:40 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 

namespace reco {
  
  typedef edm::AssociationMap<
    edm::OneToOne<reco::TrackCollection,
		  reco::TrackCollection> > TrackToTrackMap;
  
}
#endif
