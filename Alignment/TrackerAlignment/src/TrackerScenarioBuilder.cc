/// \file
///
/// $Date: 2009/03/15 17:39:58 $
/// $Revision: 1.3 $
///
/// $Author: flucke $
/// \author Frederic Ronga - CERN-PH-CMG

#include <string>
#include <iostream>
#include <sstream>

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alignment

#include "Alignment/TrackerAlignment/interface/TrackerScenarioBuilder.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"


//__________________________________________________________________________________________________
TrackerScenarioBuilder::TrackerScenarioBuilder(AlignableTracker* alignable) 
  : theAlignableTracker(alignable)
{

  if (!theAlignableTracker) {
    throw cms::Exception("TypeMismatch") << "Pointer to AlignableTracker is empty.\n";
  }

  // Fill what is needed for possiblyPartOf(..):
  theSubdets.push_back("TPB"); // Take care, order matters: 1st pixel, 2nd strip.
  theSubdets.push_back("TPE");
  theFirstStripIndex = theSubdets.size();
  theSubdets.push_back("TIB");
  theSubdets.push_back("TID");
  theSubdets.push_back("TOB");
  theSubdets.push_back("TEC");
}


//__________________________________________________________________________________________________
void TrackerScenarioBuilder::applyScenario( const edm::ParameterSet& scenario )
{

  // Apply the scenario to all main components of tracker.
  theModifierCounter = 0;

  // Seed is set at top-level, and is mandatory
  if ( this->hasParameter_( "seed", scenario) )
	theModifier.setSeed( static_cast<long>(scenario.getParameter<int>("seed")) );
  else
	throw cms::Exception("BadConfig") << "No generator seed defined!";  

  // misalignment applied recursively ('subStructures("Tracker")' contains only tracker itself)
  this->decodeMovements_(scenario, theAlignableTracker->subStructures("Tracker"));

  edm::LogInfo("TrackerScenarioBuilder") 
	<< "Applied modifications to " << theModifierCounter << " alignables";

}


//__________________________________________________________________________________________________
bool TrackerScenarioBuilder::isTopLevel_(const std::string &parameterSetName) const
{
  // Get root name (strip last character [s])
  std::string root = this->rootName_(parameterSetName);

  if (root == "Tracker") return true;

  return false;
}

//__________________________________________________________________________________________________
bool TrackerScenarioBuilder::possiblyPartOf(const std::string &subStruct, const std::string &largeStr) const
{
  // string::find(s) != nPos => 's' is contained in string!
  const std::string::size_type nPos = std::string::npos; 

  // First check whether anything from pixel in strip.
  if (largeStr.find("Strip") != nPos) {
    if (subStruct.find("Pixel") != nPos) return false;
    for (unsigned int iPix = 0; iPix < theFirstStripIndex; ++iPix) {
      if (subStruct.find(theSubdets[iPix]) != nPos) return false;
    }
  }

  // Now check whether anything from strip in pixel.
  if (largeStr.find("Pixel") != nPos) {
    if (subStruct.find("Strip") != nPos) return false;
    for (unsigned int iStrip = theFirstStripIndex; iStrip < theSubdets.size(); ++iStrip) {
      if (subStruct.find(theSubdets[iStrip]) != nPos) return false;
    }
  }

  // Finally check for any different detector parts, e.g. TIDEndcap/TIBString gives false.
  for (unsigned int iSub = 0; iSub < theSubdets.size(); ++iSub) {
    for (unsigned int iLarge = 0; iLarge < theSubdets.size(); ++iLarge) {
      if (iLarge == iSub) continue;
      if (largeStr.find(theSubdets[iLarge]) != nPos && subStruct.find(theSubdets[iSub]) != nPos) {
	return false;
      }
    }
  }

  // It seems like a possible combination:
  return true; 
}

