/// \file
///
/// $Date: 2010/11/29 20:41:55 $
/// $Revision: 1.11 $
///
/// $Author: wmtan $
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
// Call for each alignable the more general version with its appropriate level name. 
void MisalignmentScenarioBuilder::decodeMovements_(const edm::ParameterSet &pSet, 
                                                   const std::vector<Alignable*> &alignables)
{

  // first create a map with one std::vector<Alignable*> per type (=levelName)
  typedef std::map<std::string, std::vector<Alignable*> > AlignablesMap;
  AlignablesMap alisMap;
  for (std::vector<Alignable*>::const_iterator iA = alignables.begin(); iA != alignables.end(); ++iA) {
    const std::string &levelName = AlignableObjectId::idToString((*iA)->alignableObjectId());
    alisMap[levelName].push_back(*iA); // either first entry of new level or add to an old one
  }

  // Now call the more general version for each entry in the map.
  //
  // There is a hack to ensure that strip components are called in the same order
  // as in old version of TrackerScenarioBuilder (TIB,TID,TOB,TEC) while 
  // std::map seems to order alphabetically (TECEndcap,TIBHalfBarrel,TIDEndcap,TOBHalfBarrel).
  // Order matters due to random sequence. If scenarios are allowed to change
  // 'numerically', remove this comment and the lines marked with 'HACK'.
  const AlignablesMap::iterator itTec = alisMap.find("TECEndcap"); // HACK
  for (AlignablesMap::iterator it = alisMap.begin(); it != alisMap.end(); ++it) {
    if (it == itTec) continue; // HACK
    this->decodeMovements_(pSet, it->second, it->first);
  }
  if (itTec != alisMap.end()) this->decodeMovements_(pSet, itTec->second, itTec->first); // HACK
}


//__________________________________________________________________________________________________
// Decode nested parameter sets: this is the tricky part... Recursively called on components
void MisalignmentScenarioBuilder::decodeMovements_(const edm::ParameterSet &pSet, 
                                                   const std::vector<Alignable*> &alignables,
						   const std::string &levelName)
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
  for (std::vector<Alignable*>::const_iterator iter = alignables.begin();
       iter != alignables.end(); ++iter) {
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

  indent_ += " ";

  // Loop on globalSet. Add to localSet all non-existing parameters
  std::vector<std::string> globalParameterNames = globalSet.getParameterNames();
  for ( std::vector<std::string>::iterator iter = globalParameterNames.begin();
        iter != globalParameterNames.end(); iter ++ ) {

    if (globalSet.existsAs<edm::ParameterSet>(*iter)) {
      // This is a parameter set: check it
      edm::ParameterSet subLocalSet = this->getParameterSet_( (*iter), localSet );
      if ( subLocalSet.empty() ) {
        // No local subset exists: just insert it
        localSet.copyFrom(globalSet, (*iter));
      } else {
        // Merge with local subset and replace
        this->mergeParameters_( subLocalSet, globalSet.getParameter<edm::ParameterSet>(*iter) );
        localSet.addParameter<edm::ParameterSet>( (*iter), subLocalSet );
      }
    } else {
      // If (*iter) exists, (silently...) not replaced:
      localSet.copyFrom(globalSet, (*iter));
    }
  }

  indent_ = indent_.substr( 0, indent_.length()-1 );

}


//__________________________________________________________________________________________________
// Propagates some parameters from upper level.
// Parameter sets are also propagated down (if name different from global name) or merged down.
void MisalignmentScenarioBuilder::propagateParameters_( const edm::ParameterSet& pSet, 
                                                        const std::string& globalName,
                                                        edm::ParameterSet& subSet ) const
{
  indent_ += " "; // For indented output!

  // Propagate some given parameters
  std::vector<std::string> parameterNames = pSet.getParameterNames();
  for ( std::vector<std::string>::iterator iter = parameterNames.begin();
        iter != parameterNames.end(); iter++ ) {
    if ( theModifier.isPropagated( *iter ) ) { // like 'distribution', 'scale', etc.
      LogDebug("PropagateParameters") << indent_ << " - adding parameter " << (*iter) << std::endl;
      subSet.copyFrom(pSet, (*iter)); // If existing, is not replaced.
    }
  }

  // Propagate all tracked parameter sets
  std::vector<std::string> pSetNames;
  if ( pSet.getParameterSetNames( pSetNames, true ) > 0 ) {
    for ( std::vector<std::string>::const_iterator it = pSetNames.begin();
          it != pSetNames.end(); it++ ) {
      const std::string rootName = this->rootName_(*it);
      const std::string globalRoot(this->rootName_(globalName));
      if (rootName.compare(0, rootName.length(), globalRoot) == 0) {
        // Parameter for this level: skip
        LogDebug("PropagateParameters") << indent_ << " - skipping PSet " << (*it) << " from global "
					<< globalName << std::endl;
      } else if ( this->isTopLevel_(*it) ) {
        // Top-level parameters should not be propagated
	LogDebug("PropagateParameters") << indent_  
                                        << " - skipping top-level PSet " << (*it) 
                                        << " global " << globalName << std::endl;

      } else if (!this->possiblyPartOf(*it, globalRoot)) {
	// (*it) is a part of the detector that does not fit to globalName
	LogDebug("PropagateParameters") << indent_ 
                                        << " - skipping PSet " << (*it) 
					<< " not fitting into global " << globalName << std::endl;

      } else if ( AlignableObjectId::stringToId( rootName ) == align::invalid ) {
        // Parameter is not known!
        throw cms::Exception("BadConfig") << "Unknown parameter set name " << rootName;
      } else {
        // Pass down any other: in order to merge PSets, create dummy PSet
        // only containing this PSet and merge it recursively.
	LogDebug("PropagateParameters") << indent_ << " - adding PSet " << (*it) 
					<< " global " << globalName << std::endl;
        edm::ParameterSet m_subSet;
        m_subSet.addParameter<edm::ParameterSet>( (*it), 
                                                  pSet.getParameter<edm::ParameterSet>(*it) );
        this->mergeParameters_( subSet, m_subSet );
      }  
    }
  }

  indent_ = indent_.substr( 0, indent_.length()-1 );
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
    size_t lastPos = 0;
    size_t pos     = numberString.find_first_of('_', lastPos);
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
    if (showPsets || !pSet.existsAs<edm::ParameterSet>(*iter)) {
//       LogTrace("PrintParameters") << indent_ << "   " << (*iter) << " = " 
// 				  << pSet.retrieve( *iter ).toString() << std::endl;
// From Bill Tannenbaum:
// You can use
//   pset.getParameterAsString(aString).
// This function was added with the new tag.
//   However, there is a possible complication if the parameter in question is
//   itself a ParameterSet or a vector of ParameterSets.  In the new format, a
//   ParameterSet cannot be converted to a string until its ID is calculated,
//   which happens when it is registered.  So, if you get error messages about
//   not being able to convert an unregistered ParameterSet to a string, you can
//   do one of two things:
// A) You can use ParameterSet::dump() to print the parameter set, instead of
//    getParameterAsString().  This does not require registering.  I'm not sure of
//    the exact format of the dump output (Rick wrote this, I think).
// OR
// B) You can use ParameterSet::registerIt() to register the parameter set
//    before calling getParameterAsString().
//
// In either case, you can use existsAs to determine which parameters are
// themselves parameter sets or vectors of parameter sets.
//
// Note that in the new parameter set format, ParameterSet::toString() does not
// write out nested parameter sets by value.  It writes them out by
// "reference", i.e it writes the ID.
    }
  }
}


//__________________________________________________________________________________________________
bool MisalignmentScenarioBuilder::isTopLevel_( const std::string& parameterSetName ) const
{
  // Get root name (strip last character[s])
  std::string root = this->rootName_( parameterSetName );

  // tracker stuff treated in overwriting TrackerScenarioBuilder::isTopLevel_(..) 
  if ( root == "DTSector" ) return true;
  else if ( root == "CSCSector" ) return true;
  else if ( root == "Muon" ) return true;

  return false;

}

//__________________________________________________________________________________________________
bool MisalignmentScenarioBuilder::possiblyPartOf(const std::string & /*sub*/, const std::string &/*large*/) const
{
  return true; // possibly overwrite in specific class
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
