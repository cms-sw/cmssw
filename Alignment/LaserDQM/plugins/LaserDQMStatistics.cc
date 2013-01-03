/** \file LaserDQMStatistics.cc
 *  Fill the DQM Monitors
 *
 *  $Date: 2012/12/26 20:38:59 $
 *  $Revision: 1.6 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserDQM/plugins/LaserDQM.h"
#include "FWCore/Framework/interface/Event.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h" 

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

void LaserDQM::trackerStatistics(edm::Event const& theEvent,edm::EventSetup const& theSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  theSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();


  // access the tracker
  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
  theSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
  const TrackerGeometry& theTracker(*theTrackerGeometry);

  // get the StripDigiCollection
  // get the StripDigiCollection
  edm::Handle< edm::DetSetVector<SiStripDigi> > theStripDigis;
  
  for (Parameters::iterator itDigiProducersList = theDigiProducersList.begin(); itDigiProducersList != theDigiProducersList.end(); ++itDigiProducersList)
    {
      std::string digiProducer = itDigiProducersList->getParameter<std::string>("DigiProducer");
      std::string digiLabel = itDigiProducersList->getParameter<std::string>("DigiLabel");

      theEvent.getByLabel(digiProducer, digiLabel, theStripDigis);

      // loop over the entries of theStripDigis, get the DetId to identify the Detunit and find the one which will be hit by the laser beams
      for (edm::DetSetVector<SiStripDigi>::const_iterator DSViter = theStripDigis->begin(); DSViter != theStripDigis->end(); DSViter++)
	{
	  DetId theDetUnitID(DSViter->id);

	  // get the DetUnit via the DetUnitId and cast it to a StripGeomDetUnit
	  const StripGeomDetUnit * const theStripDet = dynamic_cast<const StripGeomDetUnit*>(theTracker.idToDet(theDetUnitID));

	  // get the Digis in this DetUnit
	  edm::DetSet<SiStripDigi>::const_iterator theDigiRangeIterator = (*DSViter).data.begin();
	  edm::DetSet<SiStripDigi>::const_iterator theDigiRangeIteratorEnd = (*DSViter).data.end();
      
	  // some variables we need later on in the program
	  int theBeam     = 0;
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
		
		thePart = "TIB";
		theTIBLayer = tTopo->tibLayer(theDetUnitID.rawId());
		break;
	      }
	    case StripSubdetector::TOB:
	      {
		
		thePart = "TOB";
		theTOBLayer = tTopo->tobLayer(theDetUnitID.rawId());
		theTOBStereoDet = tTopo->tobStereo(theDetUnitID.rawId());
		break;
	      }
	    case StripSubdetector::TEC:
	      {
		
	    
		// is this module in TEC+ or TEC-?
		if (tTopo->tecSide(theDetUnitID.rawId()) == 1) { thePart = "TEC-"; }
		else if (tTopo->tecSide(theDetUnitID.rawId()) == 2) { thePart = "TEC+"; }
	    
		// in which ring is this module?
		if ( theStripDet->surface().position().perp() > 55.0 && theStripDet->surface().position().perp() < 59.0 )
		  { theRing = 4; } // Ring 4
		else if ( theStripDet->surface().position().perp() > 81.0 && theStripDet->surface().position().perp() < 85.0 )
		  { theRing = 6; } // Ring 6
		else
		  { theRing = -1; } // probably not a Laser Hit!
	    
		// on which disk is this module
		theTECWheel = tTopo->tecWheel(theDetUnitID.rawId());
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


	  //       if ( ( thePart == "TEC+" || thePart == "TEC-" ) && theEvents == 1 )
	  // 	{
	  // 	  cout << " theBeam = " << theBeam << " thePart = " << thePart << " theRing = " << theRing << " Disc = " << theTECWheel << endl;
	  // 	  cout << " DetUnitId = " << theDetUnitID.rawId() << endl;
	  // 	  cout << " Phi of Det = " << theStripDet->surface().position().phi() << endl;
	  
	  // 	}

	  // fill the histograms which will be fitted at the end of the run to reconstruct the laser profile

	  /* work with else if ... for all the parts and beams */
	  // ****** beam 0 in Ring 4
	  if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 0) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam0Ring4Disc1PosAdcCounts, 
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam0Ring4Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 0 in Ring 4 ****

	  // **** Beam 1 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 1) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam1Ring4Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam1Ring4Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** TEC2TEC
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 21) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam1Ring4Disc1PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc2PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc3PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc4PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc5PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 1 in Ring 4 ****

	  // **** Beam 2 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 2) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam2Ring4Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{

		  fillAdcCounts(theMEBeam2Ring4Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam2Ring4Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  // **** Beam 2 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 22) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam2Ring4Disc1PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc2PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc3PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc4PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc5PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 2 in Ring 4 ****

	  // **** Beam 3 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 3) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam3Ring4Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc3PosAdcCounts, 
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam3Ring4Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 3 in Ring 4 ****

	  // **** Beam 4 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 4) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam4Ring4Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam4Ring4Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 24) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam4Ring4Disc1PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc2PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc3PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc4PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc5PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 4 in Ring 4 ****

	  // **** Beam 5 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 5) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam5Ring4Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam5Ring4Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 5 in Ring 4 ****

	  // **** Beam 6 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 6) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam6Ring4Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam6Ring4Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 26) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam6Ring4Disc1PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc2PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc3PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc4PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc5PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 6 in Ring 4 ****

	  // **** Beam 7 in Ring 4 ****
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 7) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam7Ring4Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam7Ring4Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  else if ( (thePart == "TEC+") && (theRing == 4) && (theBeam == 27) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam7Ring4Disc1PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc2PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc3PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc4PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc5PosTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 7 in Ring 4 ****

	  // **** Ring 6
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 0) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam0Ring6Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam0Ring6Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 0 in Ring 6 ****

	  // **** Beam 1 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 1) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam1Ring6Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam1Ring6Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 1 in Ring 6 ****

	  // **** Beam 2 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 2) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam2Ring6Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam2Ring6Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 2 in Ring 6 ****

	  // **** Beam 3 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 3) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam3Ring6Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam3Ring6Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 3 in Ring 6 ****

	  // **** Beam 4 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 4) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam4Ring6Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam4Ring6Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 4 in Ring 6 ****

	  // **** Beam 5 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 5) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam5Ring6Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam5Ring6Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 5 in Ring 6 ****

	  // **** Beam 6 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 6) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam6Ring6Disc1PosAdcCounts, 
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam6Ring6Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 6 in Ring 6 ****

	  // **** Beam 7 in Ring 6 ****
	  else if ( (thePart == "TEC+") && (theRing == 6) && (theBeam == 7) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam7Ring6Disc1PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc2PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc3PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc4PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc5PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc6PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc7PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc8PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam7Ring6Disc9PosAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 7 in Ring 6 ****

	  // ***** TEC- *****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 0) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam0Ring4Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam0Ring4Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam0Ring4Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 0 in Ring 4 ****

	  // **** Beam 1 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 1) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam1Ring4Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam1Ring4Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** TEC2TEC
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 21) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam1Ring4Disc1NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc2NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc3NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc4NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam1Ring4Disc5NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 1 in Ring 4 ****

	  // **** Beam 2 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 2) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam2Ring4Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam2Ring4Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  // **** Beam 2 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 22) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam2Ring4Disc1NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc2NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc3NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc4NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam2Ring4Disc5NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 2 in Ring 4 ****

	  // **** Beam 3 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 3) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam3Ring4Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam3Ring4Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam3Ring4Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 3 in Ring 4 ****

	  // **** Beam 4 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 4) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam4Ring4Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam4Ring4Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 24) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam4Ring4Disc1NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc2NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc3NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc4NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam4Ring4Disc5NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 4 in Ring 4 ****

	  // **** Beam 5 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 5) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam5Ring4Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam5Ring4Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam5Ring4Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 5 in Ring 4 ****

	  // **** Beam 6 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 6) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam6Ring4Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam6Ring4Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 26) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam6Ring4Disc1NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc2NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc3NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc4NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam6Ring4Disc5NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 6 in Ring 4 ****

	  // **** Beam 7 in Ring 4 ****
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 7) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam7Ring4Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam7Ring4Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // TEC2TEC
	  else if ( (thePart == "TEC-") && (theRing == 4) && (theBeam == 27) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam7Ring4Disc1NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc2NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc3NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc4NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam7Ring4Disc5NegTEC2TECAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 7 in Ring 4 ****

	  // **** Ring 6
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 0) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam0Ring6Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam0Ring6Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam0Ring6Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 0 in Ring 6 ****

	  // **** Beam 1 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 1) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam1Ring6Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam1Ring6Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam1Ring6Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 1 in Ring 6 ****

	  // **** Beam 2 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 2) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam2Ring6Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam2Ring6Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam2Ring6Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 2 in Ring 6 ****

	  // **** Beam 3 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 3) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam3Ring6Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam3Ring6Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam3Ring6Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 3 in Ring 6 ****

	  // **** Beam 4 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 4) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam4Ring6Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam4Ring6Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam4Ring6Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 4 in Ring 6 ****

	  // **** Beam 5 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 5) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam5Ring6Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam5Ring6Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam5Ring6Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 5 in Ring 6 ****

	  // **** Beam 6 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 6) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam6Ring6Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam6Ring6Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam6Ring6Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 6 in Ring 6 ****

	  // **** Beam 7 in Ring 6 ****
	  else if ( (thePart == "TEC-") && (theRing == 6) && (theBeam == 7) )
	    {
	      if ( theTECWheel == 1 )
		{ 
		  fillAdcCounts(theMEBeam7Ring6Disc1NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 2 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc2NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 3 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc3NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 4 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc4NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 5 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc5NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 6 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc6NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 7 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc7NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 8 )
		{
		  fillAdcCounts(theMEBeam7Ring6Disc8NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( theTECWheel == 9 )
		{ 
		  fillAdcCounts(theMEBeam7Ring6Disc9NegAdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of beam 7 in Ring 6 ****

	  // ***** TOB *****
	  // **** Beam 0 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 0) && (theTOBStereoDet == 0) && (theStripDet->surface().position().perp() < 58.5) )
	    {
	      if ( (theStripDet->surface().position().z() > 99.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam0TOBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 64.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam0TOBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 27.5 - theSearchZTOB) && (theStripDet->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam0TOBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -10.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam0TOBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -46.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam0TOBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -80.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam0TOBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 0 in TOB ****
      
	  // **** Beam 1 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 1) && (theTOBStereoDet == 0) && (theStripDet->surface().position().perp() < 58.5) )
	    {
	      if ( (theStripDet->surface().position().z() > 99.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam1TOBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 64.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam1TOBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 27.5 - theSearchZTOB) && (theStripDet->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam1TOBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -10.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam1TOBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -46.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam1TOBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -80.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam1TOBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 1 in TOB ****
      
	  // **** Beam 2 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 2) && (theTOBStereoDet == 0) && (theStripDet->surface().position().perp() < 58.5) )
	    {
	      if ( (theStripDet->surface().position().z() > 99.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam2TOBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 64.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam2TOBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 27.5 - theSearchZTOB) && (theStripDet->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam2TOBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -10.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam2TOBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -46.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam2TOBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -80.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam2TOBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 2 in TOB ****
      
	  // **** Beam 3 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 3) && (theTOBStereoDet == 0) && (theStripDet->surface().position().perp() < 58.5) )
	    {
	      if ( (theStripDet->surface().position().z() > 99.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam3TOBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 64.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam3TOBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 27.5 - theSearchZTOB) && (theStripDet->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam3TOBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -10.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam3TOBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -46.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam3TOBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -80.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam3TOBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 3 in TOB ****

	  // **** Beam 4 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 4) && (theTOBStereoDet == 0) && (theStripDet->surface().position().perp() < 58.5) )
	    {
	      if ( (theStripDet->surface().position().z() > 99.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam4TOBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 64.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam4TOBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 27.5 - theSearchZTOB) && (theStripDet->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam4TOBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -10.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam4TOBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -46.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam4TOBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -80.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam4TOBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 4 in TOB ****

	  // **** Beam 5 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 5) && (theTOBStereoDet == 0) && (theStripDet->surface().position().perp() < 58.5) )
	    {
	      if ( (theStripDet->surface().position().z() > 99.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam5TOBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 64.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam5TOBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 27.5 - theSearchZTOB) && (theStripDet->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam5TOBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -10.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam5TOBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -46.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam5TOBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -80.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam5TOBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 5 in TOB ****

	  // **** Beam 6 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 6) && (theTOBStereoDet == 0) && (theStripDet->surface().position().perp() < 58.5) )
	    {
	      if ( (theStripDet->surface().position().z() > 99.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam6TOBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 64.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam6TOBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 27.5 - theSearchZTOB) && (theStripDet->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam6TOBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -10.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam6TOBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -46.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam6TOBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -80.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam6TOBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 6 in TOB ****
      
	  // **** Beam 7 in TOB ****
	  else if ( (thePart == "TOB") && (theTOBLayer == 1) && (theBeam == 7) && (theTOBStereoDet == 0) && (theStripDet->surface().position().perp() < 58.5) )
	    {
	      if ( (theStripDet->surface().position().z() > 99.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 99.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam7TOBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 64.0 - theSearchZTOB) && (theStripDet->surface().position().z() < 64.0 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam7TOBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 27.5 - theSearchZTOB) && (theStripDet->surface().position().z() < 27.5 + theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam7TOBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -10.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -10.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam7TOBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -46.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -46.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam7TOBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -80.0 + theSearchZTOB) && (theStripDet->surface().position().z() > -80.0 - theSearchZTOB) )
		{ 
		  fillAdcCounts(theMEBeam7TOBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 7 in TOB ****

	  // ***** TIB *****
	  // **** Beam 0 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 0) )
	    {
	      if ( (theStripDet->surface().position().z() > 60.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam0TIBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 37.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam0TIBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 15.0 - theSearchZTIB) && (theStripDet->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam0TIBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -7.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam0TIBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -30.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam0TIBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -53.0 + theSearchZTIB) && (theStripDet->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam0TIBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 0 in TIB ****
      
	  // **** Beam 1 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 1) ) 
	    {
	      if ( (theStripDet->surface().position().z() > 60.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam1TIBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 37.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam1TIBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 15.0 - theSearchZTIB) && (theStripDet->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam1TIBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -7.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam1TIBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -30.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam1TIBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -53.0 + theSearchZTIB) && (theStripDet->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam1TIBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 1 in TIB ****
      
	  // **** Beam 2 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 2) )
	    {
	      if ( (theStripDet->surface().position().z() > 60.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam2TIBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 37.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam2TIBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 15.0 - theSearchZTIB) && (theStripDet->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam2TIBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -7.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam2TIBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -30.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam2TIBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -53.0 + theSearchZTIB) && (theStripDet->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam2TIBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 2 in TIB ****
      
	  // **** Beam 3 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 3) )
	    {
	      if ( (theStripDet->surface().position().z() > 60.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam3TIBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 37.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam3TIBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 15.0 - theSearchZTIB) && (theStripDet->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam3TIBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -7.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam3TIBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -30.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam3TIBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -53.0 + theSearchZTIB) && (theStripDet->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam3TIBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 3 in TIB ****

	  // **** Beam 4 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 4) )
	    {
	      if ( (theStripDet->surface().position().z() > 60.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam4TIBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 37.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam4TIBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 15.0 - theSearchZTIB) && (theStripDet->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam4TIBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -7.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam4TIBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -30.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam4TIBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -53.0 + theSearchZTIB) && (theStripDet->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam4TIBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 4 in TIB ****
      
	  // **** Beam 5 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 5) )
	    {
	      if ( (theStripDet->surface().position().z() > 60.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam5TIBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 37.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam5TIBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 15.0 - theSearchZTIB) && (theStripDet->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam5TIBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -7.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam5TIBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -30.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam5TIBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -53.0 + theSearchZTIB) && (theStripDet->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam5TIBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 5 in TIB ****

	  // **** Beam 6 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 6) )
	    {
	      if ( (theStripDet->surface().position().z() > 60.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam6TIBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 37.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam6TIBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 15.0 - theSearchZTIB) && (theStripDet->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam6TIBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -7.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam6TIBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -30.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam6TIBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -53.0 + theSearchZTIB) && (theStripDet->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam6TIBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 6 in TIB ****
      
	  // **** Beam 7 in TIB ****
	  else if ( (thePart == "TIB") && (theTIBLayer == 4) && (theBeam == 7) )
	    {
	      if ( (theStripDet->surface().position().z() > 60.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 60.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam7TIBPosition1AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 37.5 - theSearchZTIB) && (theStripDet->surface().position().z() < 37.5 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam7TIBPosition2AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() > 15.0 - theSearchZTIB) && (theStripDet->surface().position().z() < 15.0 + theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam7TIBPosition3AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -7.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -7.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam7TIBPosition4AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -30.5 + theSearchZTIB) && (theStripDet->surface().position().z() > -30.5 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam7TIBPosition5AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	      else if ( (theStripDet->surface().position().z() < -53.0 + theSearchZTIB) && (theStripDet->surface().position().z() > -53.0 - theSearchZTIB) )
		{ 
		  fillAdcCounts(theMEBeam7TIBPosition6AdcCounts,
				theDigiRangeIterator, theDigiRangeIteratorEnd);
		}
	    }
	  // **** end of Beam 7 in TIB ****
	}
    }
}
