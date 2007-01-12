/** \file
 *
 *  $Date: 2006/08/04 20:18:51 $
 *  $Revision: 1.5 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */
 

#include <string>
#include <iostream>
#include <sstream>

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alignment

#include "Alignment/MuonAlignment/interface/MuonScenarioBuilder.h"

//__________________________________________________________________________________________________
MuonScenarioBuilder::MuonScenarioBuilder( Alignable* alignable )
{

  theAlignableMuon = dynamic_cast<AlignableMuon*>( alignable );

  if ( !theAlignableMuon )
    throw cms::Exception("TypeMismatch") << "Argument is not an AlignableMuon";

}


//__________________________________________________________________________________________________
void MuonScenarioBuilder::applyScenario( const edm::ParameterSet& scenario )
{

  // Apply the scenario to all main components of Muon.
  theScenario = scenario;
  theModifierCounter = 0;

  // Seed is set at top-level, and is mandatory
  if ( this->hasParameter_( "seed", theScenario ) )
	theModifier.setSeed( static_cast<long>(theScenario.getParameter<int>("seed")) );
  else
	throw cms::Exception("BadConfig") << "No generator seed defined!";  
  
  // DT Barrel
  std::vector<Alignable*> dtBarrel = theAlignableMuon->DTBarrel();
  this->decodeMovements_( theScenario, dtBarrel, "DTBarrel" );

  // CSC Endcap
  std::vector<Alignable*> cscEndcaps = theAlignableMuon->CSCEndcaps();
  this->decodeMovements_( theScenario, cscEndcaps, "CSCEndcap" );


  edm::LogInfo("TrackerScenarioBuilder") 
	<< "Applied modifications to " << theModifierCounter << " alignables";

}

