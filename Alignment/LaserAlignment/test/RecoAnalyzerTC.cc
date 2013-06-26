/** \file RecoAnalyzerTC.cc
 *  function to get some information about the TrackCandidates
 *
 *  $Date: 2007/12/04 23:51:53 $
 *  $Revision: 1.4 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/test/RecoAnalyzer.h"
#include "FWCore/Framework/interface/Event.h" 
#include "FWCore/Framework/interface/ESHandle.h" 
#include "FWCore/Framework/interface/EventSetup.h" 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h" 
#include "DataFormats/GeometryVector/interface/GlobalPoint.h" 
#include "DataFormats/DetId/interface/DetId.h" 
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h" 
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h" 
#include "DataFormats/TrackReco/interface/Track.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h" 

void RecoAnalyzer::trackerTC(edm::Event const& theEvent, edm::EventSetup const& theSetup)
{
  // label of the source
  std::string src = "ckfTrackCandidates";
  std::string srcTracks = "ctfWithMaterialTracks";
  
  // access the tracker
  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
  theSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
  const TrackerGeometry& theTracker(*theTrackerGeometry);

  // get the TrackCandidate Collection
  edm::Handle<TrackCandidateCollection> theTCCollection;
  theEvent.getByLabel(src, theTCCollection );
  // get the Track Collection
  edm::Handle<reco::TrackCollection> theTrackCollection;
  theEvent.getByLabel(srcTracks, theTrackCollection);
  
  int numberOfTC = 0;

  for (TrackCandidateCollection::const_iterator i=theTCCollection->begin(); i!=theTCCollection->end();i++)
  {
    const TrackCandidate * theTC = &(*i);
    const TrackCandidate::range& recHitVec=theTC->recHits();

    std::cout << " ******* hits of TrackCandidate " << numberOfTC << " *******" << std::endl;
      // loop over the RecHits
    for (edm::OwnVector<TrackingRecHit>::const_iterator j=recHitVec.first; j!=recHitVec.second; j++)
    {    
      if ( (*j).isValid() )
      {
        GlobalPoint HitPosition = theTracker.idToDet((*j).geographicalId())->surface().toGlobal((*j).localPosition());

        std::cout << " HitPosition (x, y, z, R, phi) = " << HitPosition.x() << " " << HitPosition.y() << " " << HitPosition.z() << " " 
          << HitPosition.perp() << " " << HitPosition.phi() << std::endl;
      }
    }
    numberOfTC++;
  }

  int nTracks = 0;
  std::cout << " Number of Tracks in this event: " << theTrackCollection->size() << std::endl;
  for( reco::TrackCollection::const_iterator i = theTrackCollection->begin(); i != theTrackCollection->end(); ++i )
  {
    nTracks++;
    std::cout << " Hits in Track " << nTracks << ": " << std::endl;
    for(  trackingRecHit_iterator j = (*i).recHitsBegin(); j != (*i).recHitsEnd(); ++j )
    {
        if ( (*j)->isValid() )
        {
          GlobalPoint HitPosition = theTracker.idToDet((*j)->geographicalId())->surface().toGlobal((*j)->localPosition());

          std::cout << "   HitPosition in Track (x, y, z, R, phi) = " << HitPosition.x() << " " << HitPosition.y() << " " << HitPosition.z() << " " 
            << HitPosition.perp() << " " << HitPosition.phi() << std::endl;
        }
    }
  }

}
