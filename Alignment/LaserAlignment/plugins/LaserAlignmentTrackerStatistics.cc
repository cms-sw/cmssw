/*
 * LaserAlignmentStatistics.icc --- LAS Reconstruction Programm - Fill the histograms 
 */

#include "Alignment/LaserAlignment/plugins/LaserAlignment.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

void LaserAlignment::trackerStatistics(edm::Event const& theEvent,edm::EventSetup const& theSetup)
{
  // the DetUnits
  TrackingGeometry::DetUnitContainer theDetUnits = theTrackerGeometry->detUnits();

  // get the StripDigiCollection
  edm::Handle< edm::DetSetVector<SiStripDigi> > theStripDigis;
  
  for (Parameters::iterator itDigiProducersList = theDigiProducersList.begin(); itDigiProducersList != theDigiProducersList.end(); ++itDigiProducersList)
    {
      std::string digiProducer = itDigiProducersList->getParameter<std::string>("DigiProducer");
      std::string digiLabel = itDigiProducersList->getParameter<std::string>("DigiLabel");

      theEvent.getByLabel(digiProducer, digiLabel, theStripDigis);

      // loop over the DetUnits and identify the Detunit and find the one which will be hit by the laser beams
      // get the DetId to access the digis of the current DetUnit
      for (TrackingGeometry::DetUnitContainer::const_iterator idet = theDetUnits.begin(); idet != theDetUnits.end(); idet++)
	{
	  // the DetUnitId
 	  DetId theDetUnitID = (*idet)->geographicalId();

 	  // has this DetUnit digis?
 	  bool theDigis = false;
	  
          // get the Digis in this DetUnit
 	  edm::DetSetVector<SiStripDigi>::const_iterator DSViter = theStripDigis->find(theDetUnitID.rawId());
	  edm::DetSet<SiStripDigi>::const_iterator theDigiRangeIterator;
	  edm::DetSet<SiStripDigi>::const_iterator theDigiRangeIteratorEnd;
	  if ( DSViter != theStripDigis->end() )
	    {
	      theDigiRangeIterator = (*DSViter).data.begin();
	      theDigiRangeIteratorEnd = (*DSViter).data.end();
	      theDigis = true;
	    }
	  else { theDigis = false; }

	  // some variables we need later on in the program
	  int theBeam     = -1;
	  int theRing     = 0;
	  std::string thePart  = "";
	  int theTIBLayer = 0;
	  int theTOBLayer = 0;
	  int theTECWheel = 0;
	  int theTOBStereoDet = 0;

	  switch (theDetUnitID.subdetId())
	    {
	    case StripSubdetector::TIB:
	      {
		TIBDetId theTIBDetId(theDetUnitID.rawId());
		thePart = "TIB";
		theTIBLayer = theTIBDetId.layer();
		break;
	      }
	    case StripSubdetector::TOB:
	      {
		TOBDetId theTOBDetId(theDetUnitID.rawId());
		thePart = "TOB";
		theTOBLayer = theTOBDetId.layer();
		theTOBStereoDet = theTOBDetId.stereo();
		break;
	      }
	    case StripSubdetector::TEC:
	      {
		TECDetId theTECDetId(theDetUnitID.rawId());
	    
		// is this module in TEC+ or TEC-?
		if (theTECDetId.side() == 1) { thePart = "TEC-"; }
		else if (theTECDetId.side() == 2) { thePart = "TEC+"; }
	    
		// in which ring is this module?
		if ( (*idet)->surface().position().perp() > 55.0 && (*idet)->surface().position().perp() < 59.0 )
		  { theRing = 4; } // Ring 4
		else if ( (*idet)->surface().position().perp() > 81.0 && (*idet)->surface().position().perp() < 85.0 )
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
	      if ( ((*idet)->surface().position().phi() > 0.39 - theSearchPhiTIB) 
		   && ((*idet)->surface().position().phi() < 0.39 + theSearchPhiTIB))          { theBeam = 0; } // beam 0 

	      else if ( ((*idet)->surface().position().phi() > 1.29 - theSearchPhiTIB) 
			&& ((*idet)->surface().position().phi() < 1.29 + theSearchPhiTIB))     { theBeam = 1; } // beam 1

	      else if ( ((*idet)->surface().position().phi() > 1.85 - theSearchPhiTIB) 
			&& ((*idet)->surface().position().phi() < 1.85 + theSearchPhiTIB))     { theBeam = 2; } // beam 2

	      else if ( ((*idet)->surface().position().phi() > 2.75 - theSearchPhiTIB) 
			&& ((*idet)->surface().position().phi() < 2.75 + theSearchPhiTIB))     { theBeam = 3; } // beam 3

	      else if ( ((*idet)->surface().position().phi() > -2.59 - theSearchPhiTIB) 
			&& ((*idet)->surface().position().phi() < -2.59 + theSearchPhiTIB))    { theBeam = 4; } // beam 4

	      else if ( ((*idet)->surface().position().phi() > -2.00 - theSearchPhiTIB) 
			&& ((*idet)->surface().position().phi() < -2.00 + theSearchPhiTIB))    { theBeam = 5; } // beam 5

	      else if ( ((*idet)->surface().position().phi() > -1.10 - theSearchPhiTIB) 
			&& ((*idet)->surface().position().phi() < -1.10 + theSearchPhiTIB))    { theBeam = 6; } // beam 6

	      else if ( ((*idet)->surface().position().phi() > -0.50 - theSearchPhiTIB) 
			&& ((*idet)->surface().position().phi() < -0.50 + theSearchPhiTIB))    { theBeam = 7; } // beam 7
	      else
		{ theBeam = -1; } // probably not a Laser Hit!
	    }
	  else if ( thePart == "TOB" && theTOBLayer == 1 )
	    {
	      if ( ((*idet)->surface().position().phi() > 0.39 - theSearchPhiTOB) 
		   && ((*idet)->surface().position().phi() < 0.39 + theSearchPhiTOB))          { theBeam = 0; } // beam 0 

	      else if ( ((*idet)->surface().position().phi() > 1.29 - theSearchPhiTOB) 
			&& ((*idet)->surface().position().phi() < 1.29 + theSearchPhiTOB))     { theBeam = 1; } // beam 1

	      else if ( ((*idet)->surface().position().phi() > 1.85 - theSearchPhiTOB)
			&& ((*idet)->surface().position().phi() < 1.85 + theSearchPhiTOB))     { theBeam = 2; } // beam 2

	      else if ( ((*idet)->surface().position().phi() > 2.75 - theSearchPhiTOB)
			&& ((*idet)->surface().position().phi() < 2.75 + theSearchPhiTOB))     { theBeam = 3; } // beam 3
	  
	      else if ( ((*idet)->surface().position().phi() > -2.59 - theSearchPhiTOB)
			&& ((*idet)->surface().position().phi() < -2.59 + theSearchPhiTOB))    { theBeam = 4; } // beam 4

	      else if ( ((*idet)->surface().position().phi() > -2.00 - theSearchPhiTOB)
			&& ((*idet)->surface().position().phi() < -2.00 + theSearchPhiTOB))    { theBeam = 5; } // beam 5

	      else if ( ((*idet)->surface().position().phi() > -1.10 - theSearchPhiTOB)
			&& ((*idet)->surface().position().phi() < -1.10 + theSearchPhiTOB))    { theBeam = 6; } // beam 6

	      else if ( ((*idet)->surface().position().phi() > -0.50 - theSearchPhiTOB)
			&& ((*idet)->surface().position().phi() < -0.50 + theSearchPhiTOB))    { theBeam = 7; } // beam 7
	      else
		{ theBeam = -1; } // probably not a Laser Hit!
	    }
	  else if ( thePart == "TEC+" || thePart == "TEC-" )
	    {
	      if ( ((*idet)->surface().position().phi() > 0.39 - theSearchPhiTEC)
		   && ((*idet)->surface().position().phi() < 0.39 + theSearchPhiTEC))          { theBeam = 0; } // beam 0 

	      else if ( ((*idet)->surface().position().phi() > 1.18 - theSearchPhiTEC)
			&& ((*idet)->surface().position().phi() < 1.18 + theSearchPhiTEC))     { theBeam = 1; } // beam 1

	      else if ( ((*idet)->surface().position().phi() > 1.96 - theSearchPhiTEC)
			&& ((*idet)->surface().position().phi() < 1.96 + theSearchPhiTEC))     { theBeam = 2; } // beam 2

	      else if ( ((*idet)->surface().position().phi() > 2.74 - theSearchPhiTEC)
			&& ((*idet)->surface().position().phi() < 2.74 + theSearchPhiTEC))     { theBeam = 3; } // beam 3

	      else if ( ((*idet)->surface().position().phi() > -2.74 - theSearchPhiTEC)
			&& ((*idet)->surface().position().phi() < -2.74 + theSearchPhiTEC))    { theBeam = 4; } // beam 4

	      else if ( ((*idet)->surface().position().phi() > -1.96 - theSearchPhiTEC)
			&& ((*idet)->surface().position().phi() < -1.96 + theSearchPhiTEC))    { theBeam = 5; } // beam 5

	      else if ( ((*idet)->surface().position().phi() > -1.18 - theSearchPhiTEC)
			&& ((*idet)->surface().position().phi() < -1.18 + theSearchPhiTEC))    { theBeam = 6; } // beam 6

	      else if ( ((*idet)->surface().position().phi() > -0.39 - theSearchPhiTEC)
			&& ((*idet)->surface().position().phi() < -0.39 + theSearchPhiTEC))    { theBeam = 7; } // beam 7

	      else if ( ((*idet)->surface().position().phi() > 1.28 - theSearchPhiTEC)
			&& ((*idet)->surface().position().phi() < 1.28 + theSearchPhiTEC))     { theBeam = 21; } // beam 1 TEC2TEC

	      else if ( ((*idet)->surface().position().phi() > 1.84 - theSearchPhiTEC)
			&& ((*idet)->surface().position().phi() < 1.84 + theSearchPhiTEC))     { theBeam = 22; } // beam 2 TEC2TEC

	      else if ( ((*idet)->surface().position().phi() > -2.59 - theSearchPhiTEC)
			&& ((*idet)->surface().position().phi() < -2.59 + theSearchPhiTEC))    { theBeam = 24; } // beam 4 TEC2TEC

	      else if ( ((*idet)->surface().position().phi() > -1.10 - theSearchPhiTEC)
			&& ((*idet)->surface().position().phi() < -1.10 + theSearchPhiTEC))    { theBeam = 26; } // beam 6 TEC2TEC

	      else if ( ((*idet)->surface().position().phi() > -0.50 - theSearchPhiTEC)
			&& ((*idet)->surface().position().phi() < -0.50 + theSearchPhiTEC))    { theBeam = 27; } // beam 7 TEC2TEC
	      else 
		{ theBeam = -1; } // probably not a Laser Hit!
	    }


	  // fill the histograms which will be fitted at the end of the run to reconstruct the laser profile
	  /* work with else if ... for all the parts and beams */
	  // ****** beam 0 in Ring 4
	  if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 0) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(0)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(1)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(2)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(3)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(4)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(5)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(6)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(7)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(8)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 0 in Ring 4 ****

	  // **** Beam 1 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 1) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(9)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(10)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(11)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(12)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(13)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(14)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(15)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(16)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(17)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** TEC2TEC
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 21) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc1PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(18)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc2PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(19)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc3PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(20)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc4PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(21)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc5PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(22)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 1 in Ring 4 ****

	  // **** Beam 2 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 2) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(23)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(24)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(25)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(26)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(27)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(28)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(29)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(30)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(31)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  // **** Beam 2 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 22) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc1PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(32)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc2PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(33)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc3PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(34)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc4PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(35)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc5PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(36)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 2 in Ring 4 ****

	  // **** Beam 3 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 3) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(37)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(38)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(39)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(40)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(41)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(42)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(43)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(44)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(45)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 3 in Ring 4 ****

	  // **** Beam 4 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 4) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(46)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(47)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(48)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(49)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(50)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(51)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(52)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(53)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(54)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 24) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc1PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(55)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc2PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(56)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc3PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(57)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc4PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(58)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc5PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(59)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 4 in Ring 4 ****

	  // **** Beam 5 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 5) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(60)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(61)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(62)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(63)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(64)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(65)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(66)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(67)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(68)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 5 in Ring 4 ****

	  // **** Beam 6 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 6) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(69)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(70)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(71)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(72)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(73)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(74)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(75)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(76)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(77)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 26) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc1PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(78)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc2PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(79)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc3PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(80)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc4PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(81)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc5PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(82)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 6 in Ring 4 ****

	  // **** Beam 7 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 7) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(83)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(84)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(85)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(86)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(87)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(88)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(89)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(90)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(91)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 27) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc1PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(92)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc2PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(93)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc3PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(94)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc4PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(95)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc5PosTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(96)] = theDetIdHisto;
	      
		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 7 in Ring 4 ****

	  // **** Ring 6
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 0) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(97)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(98)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(99)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(100)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(101)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(102)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(103)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(104)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(105)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 0 in Ring 6 ****

	  // **** Beam 1 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 1) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(106)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(107)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(108)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(109)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(110)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(111)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(112)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(113)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(114)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 1 in Ring 6 ****

	  // **** Beam 2 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 2) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(115)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(116)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(117)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(118)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(119)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(120)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(121)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(122)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(123)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 2 in Ring 6 ****

	  // **** Beam 3 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 3) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(124)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(125)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(126)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(127)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(128)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(129)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(130)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(131)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(132)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 3 in Ring 6 ****

	  // **** Beam 4 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 4) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(133)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(134)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(135)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(136)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(137)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(138)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(139)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(140)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(141)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 4 in Ring 6 ****

	  // **** Beam 5 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 5) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(142)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(143)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(144)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(145)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(146)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(147)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(148)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(149)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(150)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 5 in Ring 6 ****

	  // **** Beam 6 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 6) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(151)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(152)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(153)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(154)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(155)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(156)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(157)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(158)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(159)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 6 in Ring 6 ****

	  // **** Beam 7 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 7) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc1PosAdcCounts);
		  theHistograms[theHistogramNames.at(160)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc2PosAdcCounts);
		  theHistograms[theHistogramNames.at(161)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc3PosAdcCounts);
		  theHistograms[theHistogramNames.at(162)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc4PosAdcCounts);
		  theHistograms[theHistogramNames.at(163)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc5PosAdcCounts);
		  theHistograms[theHistogramNames.at(164)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc6PosAdcCounts);
		  theHistograms[theHistogramNames.at(165)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc7PosAdcCounts);
		  theHistograms[theHistogramNames.at(166)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc8PosAdcCounts);
		  theHistograms[theHistogramNames.at(167)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc9PosAdcCounts);
		  theHistograms[theHistogramNames.at(168)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 7 in Ring 6 ****

	  // ***** TEC- *****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 0) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(169)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(170)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(171)]= theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(172)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(173)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(174)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(175)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(176)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring4Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(177)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 0 in Ring 4 ****

	  // **** Beam 1 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 1) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(178)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(179)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(180)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(181)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(182)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(183)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(184)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(185)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(186)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** TEC2TEC
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 21) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc1NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(187)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc2NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(188)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc3NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(189)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc4NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(190)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring4Disc5NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(191)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 1 in Ring 4 ****

	  // **** Beam 2 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 2) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(192)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(193)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(194)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(195)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(196)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(197)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(198)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(199)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(200)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  // **** Beam 2 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 22) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc1NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(201)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc2NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(202)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc3NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(203)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc4NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(204)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring4Disc5NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(205)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 2 in Ring 4 ****

	  // **** Beam 3 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 3) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(206)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(207)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(208)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(209)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(210)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(211)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(212)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(213)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring4Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(214)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 3 in Ring 4 ****

	  // **** Beam 4 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 4) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(215)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(216)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(217)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(218)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(219)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(220)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(221)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(222)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(223)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 24) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc1NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(224)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc2NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(225)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc3NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(226)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc4NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(227)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring4Disc5NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(228)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 4 in Ring 4 ****

	  // **** Beam 5 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 5) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(229)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(230)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(231)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(232)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(233)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(234)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(235)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(236)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring4Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(237)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 5 in Ring 4 ****

	  // **** Beam 6 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 6) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(238)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(239)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(240)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(241)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(242)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(243)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(244)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(245)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(246)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 26) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc1NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(247)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc2NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(248)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc3NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(249)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc4NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(250)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring4Disc5NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(251)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 6 in Ring 4 ****

	  // **** Beam 7 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 7) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(252)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(253)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(254)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(255)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(256)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(257)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(258)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(259)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(260)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 27) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc1NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(261)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc2NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(262)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc3NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(263)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc4NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(264)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring4Disc5NegTEC2TECAdcCounts);
		  theHistograms[theHistogramNames.at(265)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 7 in Ring 4 ****

	  // **** Ring 6
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 0) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(266)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(267)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(268)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(269)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(270)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(271)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(272)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(273)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0Ring6Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(274)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 0 in Ring 6 ****

	  // **** Beam 1 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 1) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(275)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(276)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(277)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(278)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(279)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(280)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(281)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(282)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1Ring6Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(283)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 1 in Ring 6 ****

	  // **** Beam 2 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 2) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(284)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(285)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(286)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(287)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(288)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(289)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(290)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(291)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2Ring6Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(292)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 2 in Ring 6 ****

	  // **** Beam 3 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 3) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(293)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(294)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(295)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(296)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(297)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(298)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(299)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(300)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3Ring6Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(301)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 3 in Ring 6 ****

	  // **** Beam 4 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 4) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(302)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(303)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(304)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(305)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(306)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(307)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(308)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(309)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4Ring6Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(310)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 4 in Ring 6 ****

	  // **** Beam 5 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 5) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(311)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(312)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(313)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(314)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(315)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(316)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(317)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(318)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5Ring6Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(319)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 5 in Ring 6 ****

	  // **** Beam 6 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 6) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(320)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(321)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(322)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(323)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(324)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(325)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(326)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(327)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6Ring6Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(328)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 6 in Ring 6 ****

	  // **** Beam 7 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 7) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc1NegAdcCounts);
		  theHistograms[theHistogramNames.at(329)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc2NegAdcCounts);
		  theHistograms[theHistogramNames.at(330)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc3NegAdcCounts);
		  theHistograms[theHistogramNames.at(331)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc4NegAdcCounts);
		  theHistograms[theHistogramNames.at(332)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc5NegAdcCounts);
		  theHistograms[theHistogramNames.at(333)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc6NegAdcCounts);
		  theHistograms[theHistogramNames.at(334)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc7NegAdcCounts);
		  theHistograms[theHistogramNames.at(335)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc8NegAdcCounts);
		  theHistograms[theHistogramNames.at(336)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7Ring6Disc9NegAdcCounts);
		  theHistograms[theHistogramNames.at(337)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 7 in Ring 6 ****

	  // ***** TOB *****
	  // **** Beam 0 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 0) && (theTOBStereoDet == 0) && ((*idet)->surface().position().perp() < 58.5) )
	    {
	      if ( ((*idet)->surface().position().z() > 99.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0TOBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(338)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 64.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0TOBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(339)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 27.5 - theSearchZTOB) && ((*idet)->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0TOBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(340)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -10.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0TOBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(341)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -46.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0TOBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(342)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -80.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0TOBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(343)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 0 in TOB ****
      
	  // **** Beam 1 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 1) && (theTOBStereoDet == 0) && ((*idet)->surface().position().perp() < 58.5) )
	    {
	      if ( ((*idet)->surface().position().z() > 99.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1TOBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(344)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 64.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1TOBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(345)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 27.5 - theSearchZTOB) && ((*idet)->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1TOBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(346)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -10.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1TOBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(347)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -46.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1TOBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(348)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -80.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1TOBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(349)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 1 in TOB ****
      
	  // **** Beam 2 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 2) && (theTOBStereoDet == 0) && ((*idet)->surface().position().perp() < 58.5) )
	    {
	      if ( ((*idet)->surface().position().z() > 99.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2TOBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(350)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 64.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2TOBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(351)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 27.5 - theSearchZTOB) && ((*idet)->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2TOBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(352)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -10.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2TOBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(353)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -46.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2TOBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(354)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -80.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2TOBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(355)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 2 in TOB ****
      
	  // **** Beam 3 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 3) && (theTOBStereoDet == 0) && ((*idet)->surface().position().perp() < 58.5) )
	    {
	      if ( ((*idet)->surface().position().z() > 99.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3TOBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(356)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 64.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3TOBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(357)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 27.5 - theSearchZTOB) && ((*idet)->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3TOBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(358)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -10.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3TOBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(359)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -46.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3TOBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(360)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -80.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3TOBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(361)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 3 in TOB ****

	  // **** Beam 4 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 4) && (theTOBStereoDet == 0) && ((*idet)->surface().position().perp() < 58.5) )
	    {
	      if ( ((*idet)->surface().position().z() > 99.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4TOBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(362)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 64.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4TOBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(363)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 27.5 - theSearchZTOB) && ((*idet)->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4TOBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(364)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -10.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4TOBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(365)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -46.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4TOBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(366)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -80.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4TOBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(367)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 4 in TOB ****

	  // **** Beam 5 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 5) && (theTOBStereoDet == 0) && ((*idet)->surface().position().perp() < 58.5) )
	    {
	      if ( ((*idet)->surface().position().z() > 99.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5TOBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(368)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 64.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5TOBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(369)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 27.5 - theSearchZTOB) && ((*idet)->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5TOBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(370)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -10.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5TOBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(371)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -46.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5TOBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(372)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -80.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5TOBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(373)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 5 in TOB ****

	  // **** Beam 6 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 6) && (theTOBStereoDet == 0)  && ((*idet)->surface().position().perp() < 58.5) )
	    {
	      if ( ((*idet)->surface().position().z() > 99.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6TOBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(374)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 64.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6TOBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(375)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 27.5 - theSearchZTOB) && ((*idet)->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6TOBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(376)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -10.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6TOBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(377)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -46.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6TOBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(378)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -80.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6TOBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(379)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 6 in TOB ****
      
	  // **** Beam 7 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 7) && (theTOBStereoDet == 0) && ((*idet)->surface().position().perp() < 58.5) )
	    {
	      if ( ((*idet)->surface().position().z() > 99.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7TOBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(380)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 64.0 - theSearchZTOB) && ((*idet)->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7TOBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(381)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 27.5 - theSearchZTOB) && ((*idet)->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7TOBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(382)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -10.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7TOBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(383)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -46.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7TOBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(384)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -80.0 + theSearchZTOB) && ((*idet)->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7TOBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(385)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 7 in TOB ****

	  // ***** TIB *****
	  // **** Beam 0 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 0) )
	    {
	      if ( ((*idet)->surface().position().z() > 60.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0TIBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(386)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 37.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0TIBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(387)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 15.0 - theSearchZTIB) && ((*idet)->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0TIBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(388)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -7.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0TIBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(389)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -30.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0TIBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(390)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -53.0 + theSearchZTIB) && ((*idet)->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam0TIBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(391)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 0 in TIB ****
      
	  // **** Beam 1 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 1) ) 
	    {
	      if ( ((*idet)->surface().position().z() > 60.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1TIBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(392)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 37.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1TIBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(393)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 15.0 - theSearchZTIB) && ((*idet)->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1TIBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(394)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -7.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1TIBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(395)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -30.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1TIBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(396)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -53.0 + theSearchZTIB) && ((*idet)->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam1TIBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(397)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 1 in TIB ****
      
	  // **** Beam 2 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 2) )
	    {
	      if ( ((*idet)->surface().position().z() > 60.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2TIBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(398)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 37.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2TIBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(399)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 15.0 - theSearchZTIB) && ((*idet)->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2TIBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(400)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -7.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2TIBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(401)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -30.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2TIBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(402)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -53.0 + theSearchZTIB) && ((*idet)->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam2TIBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(403)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 2 in TIB ****
      
	  // **** Beam 3 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 3) )
	    {
	      if ( ((*idet)->surface().position().z() > 60.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3TIBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(404)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 37.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3TIBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(405)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 15.0 - theSearchZTIB) && ((*idet)->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3TIBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(406)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -7.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3TIBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(407)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -30.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3TIBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(408)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -53.0 + theSearchZTIB) && ((*idet)->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam3TIBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(409)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 3 in TIB ****

	  // **** Beam 4 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 4) )
	    {
	      if ( ((*idet)->surface().position().z() > 60.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4TIBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(410)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 37.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4TIBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(411)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 15.0 - theSearchZTIB) && ((*idet)->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4TIBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(412)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -7.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4TIBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(413)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -30.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4TIBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(414)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -53.0 + theSearchZTIB) && ((*idet)->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam4TIBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(415)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 4 in TIB ****
      
	  // **** Beam 5 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 5) )
	    {
	      if ( ((*idet)->surface().position().z() > 60.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5TIBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(416)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 37.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5TIBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(417)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 15.0 - theSearchZTIB) && ((*idet)->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5TIBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(418)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -7.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5TIBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(419)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -30.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5TIBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(420)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -53.0 + theSearchZTIB) && ((*idet)->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam5TIBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(421)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 5 in TIB ****

	  // **** Beam 6 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 6) )
	    {
	      if ( ((*idet)->surface().position().z() > 60.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6TIBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(422)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 37.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6TIBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(423)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 15.0 - theSearchZTIB) && ((*idet)->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6TIBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(424)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -7.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6TIBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(425)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -30.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6TIBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(426)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -53.0 + theSearchZTIB) && ((*idet)->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam6TIBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(427)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 6 in TIB ****
      
	  // **** Beam 7 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 7) )
	    {
	      if ( ((*idet)->surface().position().z() > 60.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7TIBPosition1AdcCounts);
		  theHistograms[theHistogramNames.at(428)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 37.5 - theSearchZTIB) && ((*idet)->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7TIBPosition2AdcCounts);
		  theHistograms[theHistogramNames.at(429)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() > 15.0 - theSearchZTIB) && ((*idet)->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7TIBPosition3AdcCounts);
		  theHistograms[theHistogramNames.at(430)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -7.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7TIBPosition4AdcCounts);
		  theHistograms[theHistogramNames.at(431)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -30.5 + theSearchZTIB) && ((*idet)->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7TIBPosition5AdcCounts);
		  theHistograms[theHistogramNames.at(432)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( ((*idet)->surface().position().z() < -53.0 + theSearchZTIB) && ((*idet)->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  std::pair<DetId, TH1D*> theDetIdHisto(theDetUnitID, theBeam7TIBPosition6AdcCounts);
		  theHistograms[theHistogramNames.at(433)] = theDetIdHisto;

		  if (theDigis) fillAdcCounts(theDetIdHisto.second, theDetUnitID, theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 7 in TIB ****
	}
    }
}
