#include <string>
#include <iostream>
#include <sstream>

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alignment

#include "Alignment/MuonAlignment/interface/MisalignmentScenarioBuilder.h"


//__________________________________________________________________________________________________
void MisalignmentScenarioBuilder::applyScenario( const edm::ParameterSet& scenario )
{

  // Apply the scenario to all main components of Muon.
  theScenario = scenario;


  // Seed is set at top-level, and is mandatory
  if ( this->hasParameter_( "seed", theScenario ) )
	theMuonModifier.setSeed( static_cast<long>(theScenario.getParameter<int>("seed")) );
  else
	throw cms::Exception("BadConfig") << "No generator seed defined!";  
  
  // DT Barrel
  std::vector<Alignable*> dtBarrel = theMuon->DTBarrel();
  this->decodeMovements_( theScenario, dtBarrel, "DT" );

  // CSC Endcap
  std::vector<Alignable*> cscEndcaps = theMuon->CSCEndcaps();
  this->decodeMovements_( theScenario, cscEndcaps, "CSC" );


}


//__________________________________________________________________________________________________


//__________________________________________________________________________________________________
// Decode nested parameter sets: this is the tricky part... Recursively called on components
void MisalignmentScenarioBuilder::decodeMovements_( const edm::ParameterSet& pSet, 
						          std::vector<Alignable*> alignables,
       							  std::string levelName )
{

  indent += " "; // For indented output!

  // Retrieve parameters for all components at this level
  std::ostringstream name;
  name << levelName << "s";
  edm::ParameterSet globalParameters = this->getParameterSet_( name.str(), pSet );
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
	  edm::ParameterSet localParameters = this->getParameterSet_( name.str(), pSet );
	  LogDebug("PrintParameters") << indent << " ** " << name.str() << ": found "
								  << localParameters.getParameterNames().size() 
								  << " local parameters"  << std::endl;
	  this->mergeParameters_( localParameters, globalParameters );
	  
	  // Retrieve and apply parameters
	  LogDebug("PrintParameters")  << indent << " parameters to apply:" << std::endl;
	  this->printParameters_( localParameters, true );
	  if ( theMuonModifier.modify( (*iter), localParameters ) )
		edm::LogInfo("PrintParameters") << indent << "Movements applied to " << name.str();

	  // Apply movements to components
	  std::vector<std::string> parameterSetNames;
	  localParameters.getParameterSetNames( parameterSetNames, true );
	  if ( (*iter)->size() > 0 && parameterSetNames.size() > 0 )
		// Has components and remaining parameter sets
		this->decodeMovements_( localParameters, (*iter)->components() , levelName );
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
		  edm::ParameterSet subLocalSet = this->getParameterSet_( (*iter), localSet );
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
	if ( theMuonModifier.isPropagated( *iter ) )
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
edm::ParameterSet MisalignmentScenarioBuilder::getParameterSet_( const std::string& name,
																 const edm::ParameterSet& pSet ) const
{

  edm::ParameterSet result;

  // Get list of parameter set names and retrieve requested one
  std::vector<std::string> parameterSetNames;
  if ( this->hasParameter_( name, pSet ) )
	result = pSet.getParameter<edm::ParameterSet>( name );

  return result;

}



//__________________________________________________________________________________________________
bool MisalignmentScenarioBuilder::hasParameter_( const std::string& name,
												 const edm::ParameterSet& pSet ) const
{

  // Get list of parameter set names and look for requested one
  std::vector<std::string> names = pSet.getParameterNames();
  return ( std::find( names.begin(), names.end(), name ) != names.end() );

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
