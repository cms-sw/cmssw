/** \file RecoAnalyzerTC.cc
 *  function to get some information about the TrackCandidates
 *
 *  $Date: Sun Mar 18 19:55:59 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/test/RecoAnalyzer.h"

void RecoAnalyzer::trackerTC(edm::Event const& theEvent, edm::EventSetup const& theSetup)
{
  // label of the source
  std::string src = "ckfTrackCandidates";
  
  // access the tracker
  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
  theSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
  const TrackerGeometry& theTracker(*theTrackerGeometry);

  // get the TrackCandidate Collection
  edm::Handle<TrackCandidateCollection> theTCCollection;
  theEvent.getByLabel(src, theTCCollection );
  
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
}
