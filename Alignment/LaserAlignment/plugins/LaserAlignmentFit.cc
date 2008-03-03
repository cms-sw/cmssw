/** \file LaserAlignmentFit.cc
 *  LAS Reconstruction Program - Fitting of the Beam Profiles
 *
 *  $Date: 2007/12/04 23:51:42 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/plugins/LaserAlignment.h" 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

void LaserAlignment::fit(edm::EventSetup const& theSetup)
{
  std::cout << " theBeamProfileFitStore has " << theBeamProfileFitStore.size() << " entries " << std::endl;

  // loop over the map with the histograms and do the BeamProfileFit
  for (std::vector<std::string>::const_iterator iHistName = theHistogramNames.begin(); iHistName != theHistogramNames.end(); ++iHistName)
    {
      std::map<std::string, std::pair<DetId, TH1D*> >::iterator iHist = theHistograms.find(*iHistName);
      if ( iHist != theHistograms.end() )
	{
	  LogDebug("LaserAlignment::fit()") << " doing the fit for " << *iHistName;


	  std::string stringDisc;
	  std::string stringRing;
	  std::string stringBeam;
	  bool isTEC2TEC = false;
	  int theDisc = 0;
	  int theRing = 0;
	  int theBeam = 0;
	  int theTECSide = 0;
	  
	  // check if we are in the Endcap
	  switch (((iHist->second).first).subdetId())
	    {
	    case StripSubdetector::TIB:
	      {
		break;
	      }
	    case StripSubdetector::TOB:
	      {
		break;
	      }
	    case StripSubdetector::TEC:
	      {
		TECDetId theTECDetId(((iHist->second).first).rawId());

		theTECSide = theTECDetId.side(); // 1 for TEC-, 2 for TEC+

		stringBeam = (*iHistName).at(4);
		stringRing = (*iHistName).at(9);
		stringDisc = (*iHistName).at(14);
		isTEC2TEC = ( (*iHistName).size() > 21 ) ? true : false;
		break;
	      }
	    }

	  if ( stringRing == "4" )      { theRing = 4; }
	  else if ( stringRing == "6" ) { theRing = 6; }


	  if ( stringDisc == "1" )      { theDisc = 0; }
	  else if ( stringDisc == "2" ) { theDisc = 1; }
	  else if ( stringDisc == "3" ) { theDisc = 2; } 
	  else if ( stringDisc == "4" ) { theDisc = 3; } 
	  else if ( stringDisc == "5" ) { theDisc = 4; }
	  else if ( stringDisc == "6" ) { theDisc = 5; } 
	  else if ( stringDisc == "7" ) { theDisc = 6; } 
	  else if ( stringDisc == "8" ) { theDisc = 7; } 
	  else if ( stringDisc == "9" ) { theDisc = 8; } 

	  if ( theRing == 4 )
	    {
	      if ( stringBeam == "0" )      { theBeam = 0; } 
	      else if ( stringBeam == "1" ) { theBeam = 1; } 
	      else if ( stringBeam == "2" ) { theBeam = 2; }
	      else if ( stringBeam == "3" ) { theBeam = 3; } 
	      else if ( stringBeam == "4" ) { theBeam = 4; }
	      else if ( stringBeam == "5" ) { theBeam = 5; } 
	      else if ( stringBeam == "6" ) { theBeam = 6; } 
	      else if ( stringBeam == "7" ) { theBeam = 7; } 
	    }
	  else if ( theRing == 6 ) {
	    if ( stringBeam == "0" )      { theBeam = 0 + 8; } 
	    else if ( stringBeam == "1" ) { theBeam = 1 + 8; } 
	    else if ( stringBeam == "2" ) { theBeam = 2 + 8; }
	    else if ( stringBeam == "3" ) { theBeam = 3 + 8; } 
	    else if ( stringBeam == "4" ) { theBeam = 4 + 8; }
	    else if ( stringBeam == "5" ) { theBeam = 5 + 8; } 
	    else if ( stringBeam == "6" ) { theBeam = 6 + 8; } 
	    else if ( stringBeam == "7" ) { theBeam = 7 + 8; } 
	  }

	  std::vector<LASBeamProfileFit> collector;


	  collector = theBeamFitter->doFit(theSetup,(iHist->second).first,
					   (iHist->second).second, 
					   theSaveHistograms,
					   theNEventsPerLaserIntensity, 
					   theBeam, theDisc, theRing, 
					   theTECSide, isTEC2TEC, theIsGoodFit);

	  // if the fit succeeded, add the LASBeamProfileFit to the output collection for storage
	  // and additionally add the LASBeamProfileFit to the map for later processing (we need
	  // the info from the fit for the Alignment Algorithm)
	  if (theIsGoodFit) {
	    
	    // add the result of the fit to the map
	    theBeamProfileFitStore[*iHistName] = collector;
	  }
	  
	  // set theIsGoodFit to false again for the next fit
	  theIsGoodFit = false;
	}
      else 
	{
	  LogDebug("LaserAlignment::fit()") << " ERROR!!! can not find entry for " << (*iHistName) << " in the map ";
	}
    }

}

