/// \file
///
/// $Date: 2008/04/22 22:56:22 $
/// $Revision: 1.7 $
///
/// $Author: flucke $
/// \author Frederic Ronga - CERN-PH-CMG

#include <string>
#include <iostream>
#include <sstream>
#include <stdlib.h>

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alignment
#include "Alignment/CommonAlignment/interface/MisalignmentScenarioBuilder.h"
#include "Alignment/CommonAlignment/interface/Alignable.h" 


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

  indent_ += " "; // For indented output!

  // Retrieve parameters for all components at this level
  std::ostringstream name;
  name << levelName << "s";
  edm::ParameterSet globalParameters = this->getParameterSet_( name.str(), pSet );
  if ( !globalParameters.empty() ) {
    LogDebug("PrintParameters") << indent_ << " *** " << levelName << ": found "
                                << globalParameters.getParameterNames().size() 
                                << " global parameters" << std::endl;
  }
  
  // Propagate down parameters from upper level
  this->propagateParameters_( pSet, name.str(), globalParameters );
  LogDebug("PrintParameters") << indent_ << " global parameter is now:" << std::endl;
  this->printParameters_( globalParameters, true );

  // Loop on alignables
  int iComponent = 0; // physical numbering starts at 1...
  for ( std::vector<Alignable*>::iterator iter = alignables.begin();
        iter != alignables.end(); ++iter ) {
    iComponent++;

    // Check for special parameters -> merge with global
    name.str("");
    name << levelName << iComponent;
    edm::ParameterSet localParameters = this->getParameterSet_( levelName, iComponent, pSet );
    LogDebug("PrintParameters") << indent_ << " ** " << name.str() << ": found "
                                << localParameters.getParameterNames().size() 
                                << " local parameters"  << std::endl;
    this->mergeParameters_( localParameters, globalParameters );
	  
    // Retrieve and apply parameters
    LogDebug("PrintParameters")  << indent_ << " parameters to apply:" << std::endl;
    this->printParameters_( localParameters, true );
    if ( theModifier.modify( (*iter), localParameters ) ) {
      theModifierCounter++;
      LogDebug("PrintParameters") << indent_ << "Movements applied to " << name.str();
    }

    // Apply movements to components
    std::vector<std::string> parameterSetNames;
    localParameters.getParameterSetNames( parameterSetNames, true );
    if ( (*iter)->size() > 0 && parameterSetNames.size() > 0 )
      // Has components and remaining parameter sets
      this->decodeMovements_( localParameters, (*iter)->components() );
  }

  indent_ = indent_.substr( 0, indent_.length()-1 );

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
        iter != globalParameterNames.end(); iter ++ ) {
    if ( globalSet.retrieve( *iter ).typeCode() == 'P' ) {
      // This is a parameter set: check it
      edm::ParameterSet subLocalSet = this->getParameterSet_( (*iter), localSet );
      if ( subLocalSet.empty() ) {
        // No local subset exists: just insert it
        localSet.insert( false, (*iter), globalSet.retrieve(*iter) );
      } else {
        // Merge with local subset and replace
        this->mergeParameters_( subLocalSet, globalSet.getParameter<edm::ParameterSet>(*iter) );
        localSet.addParameter<edm::ParameterSet>( (*iter), subLocalSet );
      }
    } else {
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
        iter != parameterNames.end(); iter++ ) {
    if ( theModifier.isPropagated( *iter ) ) {
      LogDebug("PropagateParameters") << indent_ << " - adding parameter " << (*iter) << std::endl;
      subSet.insert( false, (*iter), pSet.retrieve(*iter) );
    }
  }

  // Propagate all tracked parameter sets
  std::vector<std::string> pSetNames;
  if ( pSet.getParameterSetNames( pSetNames, true ) > 0 ) {
    for ( std::vector<std::string>::const_iterator it = pSetNames.begin();
          it != pSetNames.end(); it++ ) {
      std::string rootName = this->rootName_( *it );
      if ( (*it).compare( 0, (*it).length()-1, 
                          this->rootName_(globalName) ) == 0 ) {
        // Parameter for this level: skip
        LogDebug("PropagateParameters") << indent_ << " - skipping PSet " << (*it) << std::endl;
      } else if ( this->isTopLevel_(*it) ) {
        // Top-level parameters should not be propagated
        LogDebug("PropagateParameters") << indent_ 
                                        << " - skipping top-level PSet " << (*it) << std::endl;
      } else if ( theAlignableObjectId.nameToType( rootName ) == align::invalid ) {
        // Parameter is not known!
        throw cms::Exception("BadConfig") << "Unknown parameter set name " << rootName;
      } else {
        // Pass down any other: in order to merge PSets, create dummy PSet
        // only containing this PSet and merge it recursively.
        LogDebug("PropagateParameters") << indent_ << " - adding PSet " << (*it) << std::endl;
        edm::ParameterSet m_subSet;
        m_subSet.addParameter<edm::ParameterSet>( (*it), 
                                                  pSet.getParameter<edm::ParameterSet>(*it) );
        this->mergeParameters_( subSet, m_subSet );
      }  
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
  if ( this->hasParameter_( name, pSet ) ) {
    result = pSet.getParameter<edm::ParameterSet>( name );
  }

  return result;
}

//__________________________________________________________________________________________________
// Get parameter set corresponding to given level name and number.
// Return empty parameter set if does not exist.
edm::ParameterSet MisalignmentScenarioBuilder::getParameterSet_( const std::string& levelName, int iComponent, 
                                                                 const edm::ParameterSet& pSet ) const
{
  edm::ParameterSet result;
  unsigned int nFittingPsets = 0;

  // Get list of parameter set names and look for requested one
  std::vector<std::string> pNames = pSet.getParameterNames();
  for (std::vector<std::string>::iterator iter = pNames.begin(); iter != pNames.end(); ++iter) {
    if (iter->find(levelName) != 0) continue; // parameter not starting with levelName

    const std::string numberString(*iter, levelName.size());
    //    if (numberString.empty() || numberString == "s") { // "s" only left means we have e.g. 'TOBs' 
    if (numberString.empty()) { // check on "s" not needed, see below
      continue;  // nothing left in levelName to be iComponent...
    }
    // now look for numbers (separated by '_', tolerating '__' or ending with '_')
    unsigned int lastPos = 0;
    unsigned int pos     = numberString.find_first_of('_', lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
      const std::string digit(numberString.substr(lastPos, pos - lastPos));

      bool isDigit = !digit.empty();
      for (std::string::const_iterator dIt = digit.begin(); dIt != digit.end(); ++dIt) {
        if (!isdigit(*dIt)) isDigit = false; // check all 'letters' to be a digit
      }
      if (!isDigit) {
        if (lastPos != 0) { // do not throw if e.g. after 'TOB' ('Det') you find only 's' (Unit<n>)
          throw cms::Exception("BadConfig") << "[MisalignmentScenarioBuilder::getParameterSet_] "
                                            << "Expect only numbers, separated by '_' after " 
                                            << levelName << " in " << *iter << std::endl;
        }
        break;
      }

      if (atoi(digit.c_str()) == iComponent) {
        ++nFittingPsets;
        LogDebug("getParameterSet_") << indent_ << "found " << *iter << " matching "
                                     << levelName << iComponent;
        result = pSet.getParameter<edm::ParameterSet>(*iter);
        break;
      }
      lastPos = numberString.find_first_not_of('_', pos);
      pos     = numberString.find_first_of('_', lastPos);
    }
  } // end loop on names of parameters in pSet
  
  if (nFittingPsets > 1) {
    throw cms::Exception("BadConfig") << "[MisalignmentScenarioBuilder::getParameterSet_] "
                                      << "Found " << nFittingPsets << " PSet for " 
                                      << levelName << " " << iComponent << "." << std::endl;
  }

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
        iter != parameterNames.end(); iter++ ) {
    if ( pSet.retrieve( *iter ).typeCode() != 'P' || showPsets ) {
      LogTrace("PrintParameters") << indent_ << "   " << (*iter) << " = " 
                                  << pSet.retrieve( *iter ).toString() << std::endl;
    }
  }
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
  else if ( root == "Muon" ) return true;

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
  if ( parameterSetName[lastChar] == 's' ) {
    result =  parameterSetName.substr( 0, lastChar );
  } else {
    // Otherwise, look for numbers (assumes names have no numbers inside...)
    for ( unsigned int ichar = 0; ichar<parameterSetName.length(); ichar++ ) {
      if ( isdigit(parameterSetName[ichar]) ) {
        result = parameterSetName.substr( 0, ichar );
        break; // Stop at first digit
      }
    }
  }

  LogDebug("PrintParameters") << "Name was " << parameterSetName << ", root is " << result;

  return result;

}
