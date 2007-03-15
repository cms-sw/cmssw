#ifndef DataFormats_Track_TrackToTrackMap_H
#define DataFormats_Track_TrackToTrackMap_H

/** \class TrackToTrackMap
 *  No description available.
 *
 *  $Date: 2007/03/13 09:51:58 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace reco {
  
  typedef edm::AssociationMap<
    edm::OneToOne<reco::TrackCollection,
		  reco::TrackCollection> > TrackToTrackMap;
  
}
#endif
