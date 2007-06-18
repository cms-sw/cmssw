/** \file RecoAnalyzerRecHits.cc
*  plots for RecHits
  *
  *  $Date: 2007/06/17 13:36:54 $
  *  $Revision: 1.5 $
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

      // some variables we need later on in the program
    int theBeam     = 0;
    int theRing     = 0;
    std::string thePart  = "";
    int theTIBLayer = 0;
    int theTOBLayer = 0;
    int theTECWheel = 0;
    int theTOBStereoDet = 0;

    switch ((*detId_iter).subdetId())
    {
      case StripSubdetector::TIB:
      {
        TIBDetId theTIBDetId((*detId_iter).rawId());
        thePart = "TIB";
        theTIBLayer = theTIBDetId.layer();
        break;
      }
      case StripSubdetector::TOB:
      {
        TOBDetId theTOBDetId((*detId_iter).rawId());
        thePart = "TOB";
        theTOBLayer = theTOBDetId.layer();
        theTOBStereoDet = theTOBDetId.stereo();
        break;
      }
      case StripSubdetector::TEC:
      {
        TECDetId theTECDetId((*detId_iter).rawId());

      // is this module in TEC+ or TEC-?
        if (theTECDetId.side() == 1) { thePart = "TEC-"; }
        else if (theTECDetId.side() == 2) { thePart = "TEC+"; }

      // in which ring is this module?
        if ( theStripDet->surface().position().perp() > 55.0 && theStripDet->surface().position().perp() < 59.0 )
          { theRing = 4; } // Ring 4
        else if ( theStripDet->surface().position().perp() > 81.0 && theStripDet->surface().position().perp() < 85.0 )
          { theRing = 6; } // Ring 6
        else
          { theRing = -1; } // probably not a Laser Hit!

      // on which disk is this module
        theTECWheel = theTECDetId.wheel();
        break;
      }
    }

      // which beam belongs these digis to
    if ( thePart == "TIB" && theTIBLayer == 4 )
    {
      if ( (theStripDet->surface().position().phi() > 0.39 - theSearchPhiTIB) 
        && (theStripDet->surface().position().phi() < 0.39 + theSearchPhiTIB))          { theBeam = 0; } // beam 0 

      else if ( (theStripDet->surface().position().phi() > 1.29 - theSearchPhiTIB) 
        && (theStripDet->surface().position().phi() < 1.29 + theSearchPhiTIB))     { theBeam = 1; } // beam 1

      else if ( (theStripDet->surface().position().phi() > 1.85 - theSearchPhiTIB) 
        && (theStripDet->surface().position().phi() < 1.85 + theSearchPhiTIB))     { theBeam = 2; } // beam 2

      else if ( (theStripDet->surface().position().phi() > 2.75 - theSearchPhiTIB) 
        && (theStripDet->surface().position().phi() < 2.75 + theSearchPhiTIB))     { theBeam = 3; } // beam 3

      else if ( (theStripDet->surface().position().phi() > -2.59 - theSearchPhiTIB) 
        && (theStripDet->surface().position().phi() < -2.59 + theSearchPhiTIB))    { theBeam = 4; } // beam 4

      else if ( (theStripDet->surface().position().phi() > -2.00 - theSearchPhiTIB) 
        && (theStripDet->surface().position().phi() < -2.00 + theSearchPhiTIB))    { theBeam = 5; } // beam 5

      else if ( (theStripDet->surface().position().phi() > -1.10 - theSearchPhiTIB) 
        && (theStripDet->surface().position().phi() < -1.10 + theSearchPhiTIB))    { theBeam = 6; } // beam 6

      else if ( (theStripDet->surface().position().phi() > -0.50 - theSearchPhiTIB) 
        && (theStripDet->surface().position().phi() < -0.50 + theSearchPhiTIB))    { theBeam = 7; } // beam 7
      else
        { theBeam = -1; } // probably not a Laser Hit!
    }
    else if ( thePart == "TOB" && theTOBLayer == 1 )
    {
      if ( (theStripDet->surface().position().phi() > 0.39 - theSearchPhiTOB) 
        && (theStripDet->surface().position().phi() < 0.39 + theSearchPhiTOB))          { theBeam = 0; } // beam 0 

      else if ( (theStripDet->surface().position().phi() > 1.29 - theSearchPhiTOB) 
        && (theStripDet->surface().position().phi() < 1.29 + theSearchPhiTOB))     { theBeam = 1; } // beam 1

      else if ( (theStripDet->surface().position().phi() > 1.85 - theSearchPhiTOB)
        && (theStripDet->surface().position().phi() < 1.85 + theSearchPhiTOB))     { theBeam = 2; } // beam 2

      else if ( (theStripDet->surface().position().phi() > 2.75 - theSearchPhiTOB)
        && (theStripDet->surface().position().phi() < 2.75 + theSearchPhiTOB))     { theBeam = 3; } // beam 3

      else if ( (theStripDet->surface().position().phi() > -2.59 - theSearchPhiTOB)
        && (theStripDet->surface().position().phi() < -2.59 + theSearchPhiTOB))    { theBeam = 4; } // beam 4

      else if ( (theStripDet->surface().position().phi() > -2.00 - theSearchPhiTOB)
        && (theStripDet->surface().position().phi() < -2.00 + theSearchPhiTOB))    { theBeam = 5; } // beam 5

      else if ( (theStripDet->surface().position().phi() > -1.10 - theSearchPhiTOB)
        && (theStripDet->surface().position().phi() < -1.10 + theSearchPhiTOB))    { theBeam = 6; } // beam 6

      else if ( (theStripDet->surface().position().phi() > -0.50 - theSearchPhiTOB)
        && (theStripDet->surface().position().phi() < -0.50 + theSearchPhiTOB))    { theBeam = 7; } // beam 7
      else
        { theBeam = -1; } // probably not a Laser Hit!
    }
    else if ( thePart == "TEC+" || thePart == "TEC-" )
    {
      if ( (theStripDet->surface().position().phi() > 0.39 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < 0.39 + theSearchPhiTEC))          { theBeam = 0; } // beam 0 

      else if ( (theStripDet->surface().position().phi() > 1.18 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < 1.18 + theSearchPhiTEC))     { theBeam = 1; } // beam 1

      else if ( (theStripDet->surface().position().phi() > 1.96 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < 1.96 + theSearchPhiTEC))     { theBeam = 2; } // beam 2

      else if ( (theStripDet->surface().position().phi() > 2.74 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < 2.74 + theSearchPhiTEC))     { theBeam = 3; } // beam 3

      else if ( (theStripDet->surface().position().phi() > -2.74 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < -2.74 + theSearchPhiTEC))    { theBeam = 4; } // beam 4

      else if ( (theStripDet->surface().position().phi() > -1.96 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < -1.96 + theSearchPhiTEC))    { theBeam = 5; } // beam 5

      else if ( (theStripDet->surface().position().phi() > -1.18 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < -1.18 + theSearchPhiTEC))    { theBeam = 6; } // beam 6

      else if ( (theStripDet->surface().position().phi() > -0.39 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < -0.39 + theSearchPhiTEC))    { theBeam = 7; } // beam 7

      else if ( (theStripDet->surface().position().phi() > 1.28 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < 1.28 + theSearchPhiTEC))     { theBeam = 21; } // beam 1 TEC2TEC

      else if ( (theStripDet->surface().position().phi() > 1.84 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < 1.84 + theSearchPhiTEC))     { theBeam = 22; } // beam 2 TEC2TEC

      else if ( (theStripDet->surface().position().phi() > -2.59 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < -2.59 + theSearchPhiTEC))    { theBeam = 24; } // beam 4 TEC2TEC

      else if ( (theStripDet->surface().position().phi() > -1.10 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < -1.10 + theSearchPhiTEC))    { theBeam = 26; } // beam 6 TEC2TEC

      else if ( (theStripDet->surface().position().phi() > -0.50 - theSearchPhiTEC)
        && (theStripDet->surface().position().phi() < -0.50 + theSearchPhiTEC))    { theBeam = 27; } // beam 7 TEC2TEC
      else 
        { theBeam = -1; } // probably not a Laser Hit!
    }


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

      if (thePart == "TEC+" || thePart == "TEC-")   
      {             
        double r_ = sqrt(pow(theStripDet->surface().toGlobal(rechit.localPosition()).x(),2) + pow(theStripDet->surface().toGlobal(rechit.localPosition()).y(),2));
        fillLaserBeamPlots(r_,theTECWheel,thePart,theRing,theBeam);
      }
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

      if (thePart == "TEC+" || thePart == "TEC-")   
      {             
        double r_ = sqrt(pow(theStripDet->surface().toGlobal(rechit.localPosition()).x(),2) + pow(theStripDet->surface().toGlobal(rechit.localPosition()).y(),2));
        fillLaserBeamPlots(r_,theTECWheel,thePart,theRing,theBeam);
      }

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

      if (thePart == "TEC+" || thePart == "TEC-")   
      {             
        double r_ = sqrt(pow(theStripDet->surface().toGlobal(rechit.localPosition()).x(),2) + pow(theStripDet->surface().toGlobal(rechit.localPosition()).y(),2));
        fillLaserBeamPlots(r_,theTECWheel,thePart,theRing,theBeam);
      }
    }
  }
}
