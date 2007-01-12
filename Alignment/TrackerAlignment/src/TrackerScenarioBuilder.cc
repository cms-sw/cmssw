/// \file
///
/// $Date$
/// $Revision$
///
/// $Author$
/// \author Frederic Ronga - CERN-PH-CMG

#include <string>
#include <iostream>
#include <sstream>

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alignment

#include "Alignment/TrackerAlignment/interface/TrackerScenarioBuilder.h"


//__________________________________________________________________________________________________
TrackerScenarioBuilder::TrackerScenarioBuilder( Alignable* alignable )
{

  theAlignableTracker = dynamic_cast<AlignableTracker*>( alignable );
 
  if ( !theAlignableTracker )
    throw cms::Exception("TypeMismatch") << "Argument is not an AlignableTracker";

}


//__________________________________________________________________________________________________
void TrackerScenarioBuilder::applyScenario( const edm::ParameterSet& scenario )
{

  // Apply the scenario to all main components of tracker.
  theScenario = scenario;
  theModifierCounter = 0;

  // Seed is set at top-level, and is mandatory
  if ( this->hasParameter_( "seed", theScenario ) )
	theModifier.setSeed( static_cast<long>(theScenario.getParameter<int>("seed")) );
  else
	throw cms::Exception("BadConfig") << "No generator seed defined!";  
  
  // TPB
  if ( theScenario.getUntrackedParameter<bool>( "misalignTPB",true) && 
	   !theScenario.getUntrackedParameter<bool>("fixTPB",     false) )
	{
	  std::vector<Alignable*> pixelBarrels = theAlignableTracker->pixelHalfBarrels();
	  this->decodeMovements_( theScenario, pixelBarrels, "TPB" );
	}

  // TPE
  if ( theScenario.getUntrackedParameter<bool>( "misalignTPE",true) && 
	   !theScenario.getUntrackedParameter<bool>("fixTPE",     false) )
	{
	  std::vector<Alignable*> pixelEndcaps = theAlignableTracker->pixelEndCaps();
	  this->decodeMovements_( theScenario, pixelEndcaps, "TPE" );
	}

  // TIB
  if ( theScenario.getUntrackedParameter<bool>( "misalignTIB",true) && 
	   !theScenario.getUntrackedParameter<bool>("fixTIB",     false) )
	{
	  std::vector<Alignable*> innerBarrels = theAlignableTracker->innerHalfBarrels();
	  this->decodeMovements_( theScenario, innerBarrels, "TIB" );
	}

  // TID
  if ( theScenario.getUntrackedParameter<bool>( "misalignTID",true) && 
	   !theScenario.getUntrackedParameter<bool>("fixTID",     false) )
	{
	  std::vector<Alignable*> innerDisks   = theAlignableTracker->TIDs();
	  this->decodeMovements_( theScenario, innerDisks, "TID" );
	}

  // TOB
  if ( theScenario.getUntrackedParameter<bool>( "misalignTOB",true) && 
	   !theScenario.getUntrackedParameter<bool>("fixTOB",     false) )
	{
	  std::vector<Alignable*> outerBarrels = theAlignableTracker->outerHalfBarrels();
	  this->decodeMovements_( theScenario, outerBarrels, "TOB" );
	}

  // TEC
  if ( theScenario.getUntrackedParameter<bool>( "misalignTEC",true) && 
	   !theScenario.getUntrackedParameter<bool>("fixTEC",     false) )
	{
	  std::vector<Alignable*> endcaps = theAlignableTracker->endCaps();
	  this->decodeMovements_( theScenario, endcaps, "TEC" );
	}

  edm::LogInfo("TrackerScenarioBuilder") 
	<< "Applied modifications to " << theModifierCounter << " alignables";

}


