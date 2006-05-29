#include <string>
#include <iostream>
#include <sstream>

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alignment
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "Alignment/TrackerAlignment/interface/MisalignmentScenarioBuilder.h"


//__________________________________________________________________________________________________
MisalignmentScenarioBuilder::MisalignmentScenarioBuilder( const edm::ParameterSet& scenario,
														  AlignableTracker* tracker ):
  theTracker( tracker ),
  theScenario( scenario )
{

}




//__________________________________________________________________________________________________
void MisalignmentScenarioBuilder::applyScenario( void )
{

  // Apply the scenario defined in theScenario to all main components of tracker.
  
  // TOB
  std::vector<Alignable*> outerBarrels = theTracker->outerHalfBarrels();
  this->decodeMovements_( theScenario, outerBarrels, "TOB" );

  // TIB
  std::vector<Alignable*> innerBarrels = theTracker->innerHalfBarrels();
  this->decodeMovements_( theScenario, innerBarrels, "TIB" );

  // TPB
  std::vector<Alignable*> pixelBarrels = theTracker->pixelHalfBarrels();
  this->decodeMovements_( theScenario, pixelBarrels, "TPB" );

  // TEC
  std::vector<Alignable*> endcaps = theTracker->endCaps();
  this->decodeMovements_( theScenario, endcaps, "TEC" );

  // TPE
  std::vector<Alignable*> pixelEndcaps = theTracker->pixelEndCaps();
  this->decodeMovements_( theScenario, pixelEndcaps, "TPE" );

  // TID
  std::vector<Alignable*> innerDisks   = theTracker->TIDs();
  this->decodeMovements_( theScenario, innerDisks, "TID" );

}


//__________________________________________________________________________________________________
// Gets the level name from the first alignable and hands over to the more general version
void MisalignmentScenarioBuilder::decodeMovements_( const edm::ParameterSet& pSet, 
												   std::vector<Alignable*> alignables )
{

  // Get name from first element
  TrackerAlignableId converter;
  std::string levelName = converter.alignableTypeName( alignables.front() );
  this->decodeMovements_( pSet, alignables, levelName );


}


//__________________________________________________________________________________________________
void MisalignmentScenarioBuilder::decodeMovements_( const edm::ParameterSet& pSet, 
													std::vector<Alignable*> alignables,
													std::string levelName )
{

  indent += " "; // For indented output...

  // Retrieve parameters for all components at this level
  std::ostringstream name;
  name << levelName << "s";
  edm::ParameterSet globalParameters = this->getParameterSet_( pSet, name.str() );
  if ( !globalParameters.empty() )
	LogDebug("PrintParameters") << indent << " *** " << levelName << ": found "
								<< globalParameters.getParameterNames().size() 
								<< " global parameters" << std::endl;
  
  // Propagate down parameters from upper level
  this->propagateParameters_( pSet, name.str(), globalParameters );
  LogDebug("PrintParameters") << indent << " global parameter is now:" << std::endl;
  this->printParameters_( globalParameters, true );

  // Loop on alignables
  int iComponent = 0; // physical numbering starts at 1...
  for ( std::vector<Alignable*>::iterator iter = alignables.begin();
		iter != alignables.end(); ++iter )
	{
	  iComponent++;
	  // Check for special parameters -> merge with global
	  name.str("");
	  name << levelName << iComponent;
	  edm::ParameterSet localParameters = this->getParameterSet_( pSet, name.str() );
	  LogDebug("PrintParameters") << indent << " ** " << name.str() << ": found "
								  << localParameters.getParameterNames().size() 
								  << " local parameters"  << std::endl;
	  this->mergeParameters_( localParameters, globalParameters );
	  
	  // Retrieve and apply parameters
	  LogDebug("PrintParameters")  << indent << " parameters to apply:" << std::endl;
	  this->printParameters_( localParameters, true );
	  theTrackerModifier.modify( (*iter), localParameters );

	  // Apply movements to components
	  std::vector<std::string> parameterSetNames;
	  localParameters.getParameterSetNames( parameterSetNames, true );
	  if ( (*iter)->size() > 0 && parameterSetNames.size() > 0 )
		// Has components and remaining parameter sets
		this->decodeMovements_( localParameters, (*iter)->components() );
// 	  else if ( parameterSetNames.size() > 0 )
// 		{
// 		  // Has no components: remaining parameter sets are unkown!
// 		  std::ostringstream error;
// 		  error <<  "Unknown parameter set name(s): ";
// 		  for ( std::vector<std::string>::iterator iName = parameterSetNames.begin();
// 				iName != parameterSetNames.end(); iName++ )
// 			error << " " << (*iName);
// 		  error << std::endl;
// 		  throw cms::Exception("BadConfig") << error.str();
// 		}
	}

  indent = indent.substr( 0, indent.length()-1 );

}



//__________________________________________________________________________________________________
// Merge two sets of parameters into one. The local set overrides the global one
// A recursive merging is done on parameter sets.
void MisalignmentScenarioBuilder::mergeParameters_( edm::ParameterSet& localSet, 
													const edm::ParameterSet& globalSet ) const
{

  // Loop on globalSet. Add to localSet all non-existing parameters
  std::vector<std::string> globalParameterNames = globalSet.getParameterNames();
  for ( std::vector<std::string>::iterator iter = globalParameterNames.begin();
		iter != globalParameterNames.end(); iter ++ )
	{
	  if ( globalSet.retrieve( *iter ).typeCode() == 'P' )
		{
		  // This is a parameter set: check it
		  edm::ParameterSet subLocalSet = this->getParameterSet_( localSet, (*iter) );
		  if ( subLocalSet.empty() ) 
			{
			  // No local subset exists: just insert it
			  localSet.insert( false, (*iter), globalSet.retrieve(*iter) );
			}
		  else
			{
			  // Merge with local subset and replace
			  this->mergeParameters_( subLocalSet, globalSet.getParameter<edm::ParameterSet>(*iter) );
			  localSet.addParameter<edm::ParameterSet>( (*iter), subLocalSet );
			}
		} 
	  else
		{
		  localSet.insert( false, (*iter), globalSet.retrieve(*iter) );
		}
	}

}


//__________________________________________________________________________________________________
// Propagates some parameters from upper level.
// Parameter sets are also propagated down (if name different from global name) or merged down.
void MisalignmentScenarioBuilder::propagateParameters_( const edm::ParameterSet& pSet, 
														const std::string& globalName,
														edm::ParameterSet& subSet ) const
{

  // Propagate some given parameters
  std::vector<std::string> parameterNames = pSet.getParameterNames();
  for ( std::vector<std::string>::iterator iter = parameterNames.begin();
		iter != parameterNames.end(); iter++ )
	if ( theTrackerModifier.isPropagated( *iter ) )
	  {
		LogDebug("PropagateParameters") << indent << " - adding parameter " << (*iter) << std::endl;
		subSet.insert( false, (*iter), pSet.retrieve(*iter) );
	  }
	  

  // Propagate all tracked parameter sets
  std::vector<std::string> pSetNames;
  if ( pSet.getParameterSetNames( pSetNames, true ) > 0 )
	for ( std::vector<std::string>::const_iterator it = pSetNames.begin();
		  it != pSetNames.end(); it++ )
	  if ( (*it).compare( 0, (*it).length()-1, 
						  globalName.substr(0,globalName.length()-1) ) == 0 )
		{
		  // Parameter for this level: skip
		  LogDebug("PropagateParameters") << indent << " - skipping PSet " << (*it) << std::endl;
		}
	  else if ( this->isTopLevel_(*it) )
		{
		  // Top-level parameters should not be propagated
		  LogDebug("PropagateParameters") << indent 
										  << " - skipping top-level PSet " << (*it) << std::endl;
		}
	  else
		{
		  // Pass down any other: in order to merge PSets, create dummy PSet
		  // only containing this PSet and merge it recursively.
		  LogDebug("PropagateParameters") << indent << " - adding PSet " << (*it) << std::endl;
		  edm::ParameterSet m_subSet;
		  m_subSet.addParameter<edm::ParameterSet>( (*it), pSet.getParameter<edm::ParameterSet>(*it) );
		  this->mergeParameters_( subSet, m_subSet );
		}  

}


//__________________________________________________________________________________________________
// Get parameter set corresponding to given name.
// Return empty parameter set if does not exist.
edm::ParameterSet MisalignmentScenarioBuilder::getParameterSet_( const edm::ParameterSet& pSet, 
																 const std::string& name ) const
{

  edm::ParameterSet result;

  // Get list of parameter set names and retrieve requested one
  std::vector<std::string> parameterSetNames;
  if ( pSet.getParameterSetNames( parameterSetNames, true ) > 0 )
	for ( std::vector<std::string>::iterator iter = parameterSetNames.begin();
		  iter != parameterSetNames.end(); iter++ )
	  if ( (*iter) == name )
		result = pSet.getParameter<edm::ParameterSet>( name );

  return result;

}


//__________________________________________________________________________________________________
// Print parameter set. If showPsets is 'false', do not print PSets
void MisalignmentScenarioBuilder::printParameters_( const edm::ParameterSet& pSet, 
													const bool showPsets ) const
{

  std::vector<std::string> parameterNames = pSet.getParameterNames();
  for ( std::vector<std::string>::iterator iter = parameterNames.begin();
		iter != parameterNames.end(); iter++ )
	if ( pSet.retrieve( *iter ).typeCode() != 'P' || showPsets )
	  LogDebug("PrintParameters") << indent << "   " << (*iter) << " = " 
				<< pSet.retrieve( *iter ).toString() << std::endl;

}


//__________________________________________________________________________________________________
const bool MisalignmentScenarioBuilder::isTopLevel_( const std::string& parameterSetName ) const
{

  // Get root name (strip last character)
  std::string root = parameterSetName.substr(0, parameterSetName.length()-1 );
  if      ( root == "TOB" ) return true;
  else if ( root == "TIB" ) return true;
  else if ( root == "TPB" ) return true;
  else if ( root == "TEC" ) return true;
  else if ( root == "TID" ) return true;
  else if ( root == "TPE" ) return true;

  return false;

}
