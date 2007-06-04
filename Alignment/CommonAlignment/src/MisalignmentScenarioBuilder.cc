/// \file
///
/// $Date: 2007/01/12 09:47:40 $
/// $Revision: 1.1 $
///
/// $Author: fronga $
/// \author Frederic Ronga - CERN-PH-CMG

#include <string>
#include <iostream>
#include <sstream>

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alignment
#include "Alignment/CommonAlignment/interface/MisalignmentScenarioBuilder.h"


//__________________________________________________________________________________________________
// Gets the level name from the first alignable and hands over to the more general version
void MisalignmentScenarioBuilder::decodeMovements_( const edm::ParameterSet& pSet, 
												   std::vector<Alignable*> alignables )
{

  // Get name from first element
  std::string levelName = theAlignableObjectId.typeToName( alignables.front()->alignableObjectId() );
  this->decodeMovements_( pSet, alignables, levelName );

}


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
	  if ( theModifier.modify( (*iter), localParameters ) )
		{
		  theModifierCounter++;
		  LogDebug("PrintParameters") << indent << "Movements applied to " << name.str();
		}

	  // Apply movements to components
	  std::vector<std::string> parameterSetNames;
	  localParameters.getParameterSetNames( parameterSetNames, true );
	  if ( (*iter)->size() > 0 && parameterSetNames.size() > 0 )
		// Has components and remaining parameter sets
		this->decodeMovements_( localParameters, (*iter)->components() );
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
	if ( theModifier.isPropagated( *iter ) )
	  {
		LogDebug("PropagateParameters") << indent << " - adding parameter " << (*iter) << std::endl;
		subSet.insert( false, (*iter), pSet.retrieve(*iter) );
	  }
	  

  // Propagate all tracked parameter sets
  std::vector<std::string> pSetNames;
  if ( pSet.getParameterSetNames( pSetNames, true ) > 0 )
	for ( std::vector<std::string>::const_iterator it = pSetNames.begin();
		  it != pSetNames.end(); it++ )
      {
        std::string rootName = this->rootName_( *it );
        if ( (*it).compare( 0, (*it).length()-1, 
                            this->rootName_(globalName) ) == 0 )
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
        else if ( theAlignableObjectId.nameToType( rootName ) == AlignableObjectId::invalid )
          {
            // Parameter is not known!
            throw cms::Exception("BadConfig") << "Unknown parameter set name " << rootName;
          }
        else
          {
            // Pass down any other: in order to merge PSets, create dummy PSet
            // only containing this PSet and merge it recursively.
            LogDebug("PropagateParameters") << indent << " - adding PSet " << (*it) << std::endl;
            edm::ParameterSet m_subSet;
            m_subSet.addParameter<edm::ParameterSet>( (*it), 
                                                      pSet.getParameter<edm::ParameterSet>(*it) );
            this->mergeParameters_( subSet, m_subSet );
          }  
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
  std::string root = this->rootName_( parameterSetName );
  if      ( root == "TOB" ) return true;
  else if ( root == "TIB" ) return true;
  else if ( root == "TPB" ) return true;
  else if ( root == "TEC" ) return true;
  else if ( root == "TID" ) return true;
  else if ( root == "TPE" ) return true;
  else if ( root == "DTSector" ) return true;
  else if ( root == "CSCSector" ) return true;

  return false;

}

//__________________________________________________________________________________________________
// Get root name of parameter set (e.g. return 'Rod' from 'Rods' or 'Rod1')
const std::string 
MisalignmentScenarioBuilder::rootName_( const std::string& parameterSetName ) const
{

  std::string result = parameterSetName; // Initialise to full string
  
  // Check if string ends with 's'
  const int lastChar = parameterSetName.length()-1;
  if ( parameterSetName[lastChar] == 's' ) 
      result =  parameterSetName.substr( 0, lastChar );
  else
    // Otherwise, look for numbers (assumes names have no numbers inside...)
    for ( unsigned int ichar = 0; ichar<parameterSetName.length(); ichar++ )
      if ( isdigit(parameterSetName[ichar]) )
        {
          result = parameterSetName.substr( 0, ichar );
          break; // Stop at first digit
        }

  LogDebug("PrintParameters") << "Name was " << parameterSetName << ", root is " << result;

  return result;

}
