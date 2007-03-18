/** \file RecoAnalyzerRecHits.cc
 *  plots for RecHits
 *
 *  $Date: Sun Mar 18 19:55:35 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/test/RecoAnalyzer.h"

void RecoAnalyzer::trackerRecHits(edm::Event const& theEvent, edm::EventSetup const& theSetup)
{
  // access the Tracker
  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
  theSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
  const TrackerGeometry& theTracker(*theTrackerGeometry);

  edm::Handle<SiStripMatchedRecHit2DCollection> rechitsMatchedHandle;
  edm::Handle<SiStripRecHit2DCollection> rechitsRPhiHandle;
  edm::Handle<SiStripRecHit2DCollection> rechitsStereoHandle;

  // get the RecHits from the event
  theEvent.getByLabel(theRecHitProducer,"matchedRecHit", rechitsMatchedHandle);
  theEvent.getByLabel(theRecHitProducer,"rphiRecHit", rechitsRPhiHandle);
  theEvent.getByLabel(theRecHitProducer,"stereoRecHit", rechitsStereoHandle);

  // the RecHit collections
  const SiStripMatchedRecHit2DCollection * theMatchedRecHitCollection = rechitsMatchedHandle.product();
  const SiStripRecHit2DCollection * theRPhiRecHitCollection = rechitsRPhiHandle.product();
  const SiStripRecHit2DCollection * theStereoRecHitCollection = rechitsStereoHandle.product();

  // get the detIds
  const std::vector<DetId> rhMatchedIds = theMatchedRecHitCollection->ids();
  const std::vector<DetId> rhRPhiIds = theRPhiRecHitCollection->ids();
  const std::vector<DetId> rhStereoIds = theStereoRecHitCollection->ids();

  // loop over the detIds for each RecHit Collection
  for ( std::vector<DetId>::const_iterator detId_iter = rhMatchedIds.begin(); detId_iter != rhMatchedIds.end(); detId_iter++ )
    {
      // get the DetUnit
      const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>(theTracker.idToDet((*detId_iter)));

      // get the RecHits
      SiStripMatchedRecHit2DCollection::range rechitRange = theMatchedRecHitCollection->get((*detId_iter));
      SiStripMatchedRecHit2DCollection::const_iterator rechitRangeIteratorBegin = rechitRange.first;
      SiStripMatchedRecHit2DCollection::const_iterator rechitRangeIteratorEnd = rechitRange.second;
      SiStripMatchedRecHit2DCollection::const_iterator iRecHit = rechitRangeIteratorBegin;
      // loop on the RecHits
      for (; iRecHit != rechitRangeIteratorEnd; iRecHit++)
	{
	  SiStripMatchedRecHit2D const rechit = *iRecHit;
	  theRecHitPositionsX->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).x());
	  theRecHitPositionsY->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).y());
	  theRecHitPositionsZ->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).z());
	  theRecHitPositionsYvsX->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).x(),
				       theStripDet->surface().toGlobal(rechit.localPosition()).y());
	  theRecHitPositionsPhivsZ->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).z(),
					 theStripDet->surface().toGlobal(rechit.localPosition()).phi());
	  theRecHitPositionsRvsZ->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).z(),
				       theStripDet->surface().toGlobal(rechit.localPosition()).perp());
	}
    }

  for ( std::vector<DetId>::const_iterator detId_iter = rhRPhiIds.begin(); detId_iter != rhRPhiIds.end(); detId_iter++ )
    {
      // get the DetUnit
      const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>(theTracker.idToDet((*detId_iter)));

      // get the RecHits
      SiStripRecHit2DCollection::range rechitRange = theRPhiRecHitCollection->get((*detId_iter));
      SiStripRecHit2DCollection::const_iterator rechitRangeIteratorBegin = rechitRange.first;
      SiStripRecHit2DCollection::const_iterator rechitRangeIteratorEnd = rechitRange.second;
      SiStripRecHit2DCollection::const_iterator iRecHit = rechitRangeIteratorBegin;
      // loop on the RecHits
      for (; iRecHit != rechitRangeIteratorEnd; iRecHit++)
	{
	  SiStripRecHit2D const rechit = *iRecHit;
	  theRecHitPositionsX->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).x());
	  theRecHitPositionsY->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).y());
	  theRecHitPositionsZ->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).z());
	  theRecHitPositionsYvsX->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).x(),
				       theStripDet->surface().toGlobal(rechit.localPosition()).y());
	  theRecHitPositionsPhivsZ->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).z(),
					 theStripDet->surface().toGlobal(rechit.localPosition()).phi());
	  theRecHitPositionsRvsZ->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).z(),
				      theStripDet->surface().toGlobal(rechit.localPosition()).perp());
	}
    }

  for ( std::vector<DetId>::const_iterator detId_iter = rhStereoIds.begin(); detId_iter != rhStereoIds.end(); detId_iter++ )
    {
      // get the DetUnit
      const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>(theTracker.idToDet((*detId_iter)));

      // get the RecHits
      SiStripRecHit2DCollection::range rechitRange = theStereoRecHitCollection->get((*detId_iter));
      SiStripRecHit2DCollection::const_iterator rechitRangeIteratorBegin = rechitRange.first;
      SiStripRecHit2DCollection::const_iterator rechitRangeIteratorEnd = rechitRange.second;
      SiStripRecHit2DCollection::const_iterator iRecHit = rechitRangeIteratorBegin;
      // loop on the RecHits
      for (; iRecHit != rechitRangeIteratorEnd; iRecHit++)
	{
	  SiStripRecHit2D const rechit = *iRecHit;
	  theRecHitPositionsX->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).x());
	  theRecHitPositionsY->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).y());
	  theRecHitPositionsZ->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).z());
	  theRecHitPositionsYvsX->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).x(),
				       theStripDet->surface().toGlobal(rechit.localPosition()).y());
	  theRecHitPositionsPhivsZ->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).z(),
					 theStripDet->surface().toGlobal(rechit.localPosition()).phi());
	  theRecHitPositionsRvsZ->Fill(theStripDet->surface().toGlobal(rechit.localPosition()).z(),
				       theStripDet->surface().toGlobal(rechit.localPosition()).phi());
	    
	}
    }
}
