//
// $Id: TriggerObjectStandAlone.cc,v 1.9 2011/02/02 17:06:24 vadler Exp $
//

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"

#include <boost/algorithm/string.hpp>
#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace pat;


// Const data members' definitions


const char TriggerObjectStandAlone::wildcard_;


// Private methods


// Checks a string vector for occurence of a certain string, incl. wild-card mechanism
bool TriggerObjectStandAlone::hasAnyName( const std::string & name, const std::vector< std::string > & nameVec ) const
{
  // Special cases first
  // Always false for empty vector to check
  if ( nameVec.empty() ) return false;
  // Always true for general wild-card(s)
  if ( name.find_first_not_of( wildcard_ ) == std::string::npos ) return true;
  // Split name to evaluate in parts, seperated by wild-cards
  std::vector< std::string > namePartsVec;
  boost::split( namePartsVec, name, boost::is_any_of( std::string( 1, wildcard_ ) ), boost::token_compress_on );
  // Iterate over vector of names to search
  for ( std::vector< std::string >::const_iterator iVec = nameVec.begin(); iVec != nameVec.end(); ++iVec ) {
    // Not failed yet
    bool failed( false );
    // Start searching at the first character
    size_type index( 0 );
    // Iterate over evaluation name parts
    for ( std::vector< std::string >::const_iterator iName = namePartsVec.begin(); iName != namePartsVec.end(); ++iName ) {
      // Empty parts due to
      // - wild-card at beginning/end or
      // - multiple wild-cards (should be supressed by 'boost::token_compress_on')
      if ( iName->length() == 0 ) continue;
      // Search from current index and
      // set index to found occurence
      index = iVec->find( *iName, index );
      // Failed and exit loop, if
      // - part not found
      // - part at beginning not found there
      if ( index == std::string::npos || ( iName == namePartsVec.begin() && index > 0 ) ) {
        failed = true;
        break;
      }
      // Increase index by length of found part
      index += iName->length();
    }
    // Failed, if end of name not reached
    if ( index < iVec->length() && namePartsVec.back().length() != 0 ) failed = true;
    // Match found!
    if ( ! failed ) return true;
  }
  // No match found!
  return false;
}


// Adds a new HLT path or L1 algorithm name
void TriggerObjectStandAlone::addPathOrAlgorithm( const std::string & name, bool firing )
{
  // Check, if path is already assigned
  if ( ! hasPathOrAlgorithm( name, false ) ) {
    // The path itself
    pathNames_.push_back( name );
    // The corresponding usage of the trigger objects
    pathLastFilterAccepted_.push_back( firing );
  }
}


// Gets all HLT path or L1 algorithm names
std::vector< std::string > TriggerObjectStandAlone::pathsOrAlgorithms( bool firing ) const
{
  // All path names, if usage not restricted (not required or not available)
  if ( ! firing || ! hasFiring() ) return pathNames_;
  // Temp vector of path names
  std::vector< std::string > paths;
  // Loop over usage vector and fill corresponding paths into temp vector
  for ( unsigned iPath = 0; iPath < pathNames_.size(); ++iPath ) {
    if ( pathLastFilterAccepted_.at( iPath ) ) paths.push_back( pathNames_.at( iPath ) );
  }
  // Return temp vector
  return paths;
}


// Checks, if a certain HLT filter label or L1 condition name is assigned
bool TriggerObjectStandAlone::hasFilterOrCondition( const std::string & name ) const
{
  // Move to wild-card parser, if needed
  if ( name.find( wildcard_ ) != std::string::npos ) return hasAnyName( name, filterLabels_ );
  // Return, if filter label is assigned
  return ( std::find( filterLabels_.begin(), filterLabels_.end(), name ) != filterLabels_.end() );
}


// Checks, if a certain path name is assigned
bool TriggerObjectStandAlone::hasPathOrAlgorithm( const std::string & name, bool firing ) const
{
  // Move to wild-card parser, if needed
  if ( name.find( wildcard_ ) != std::string::npos ) return hasAnyName( name, pathsOrAlgorithms( firing ) );
  // Deal with older PAT-tuples, where trigger object usage is not available
  if ( ! hasFiring() ) firing = false;
  // Check, if path name is assigned at all
  std::vector< std::string >::const_iterator match( std::find( pathNames_.begin(), pathNames_.end(), name ) );
  // False, if path name not assigned
  if ( match == pathNames_.end() ) return false;
  // Return for assigned path name, if trigger object usage meets requirement
  return ( firing ? pathLastFilterAccepted_.at( match - pathNames_.begin() ) : true );
}


// Methods


// Gets the pat::TriggerObject (parent class)
TriggerObject TriggerObjectStandAlone::triggerObject()
{
  // Create a TriggerObjects
  TriggerObject theObj( p4(), pdgId() );
  // Set its collection and trigger objects types (no c'tor for that)
  theObj.setCollection( collection() );
  for ( size_t i = 0; i < triggerObjectTypes().size(); ++i ) theObj.addTriggerObjectType( triggerObjectTypes().at( i ) );
  // Return TriggerObject
  return theObj;
}


// Checks, if a certain label of original collection is assigned (method overrides)
bool TriggerObjectStandAlone::hasCollection( const std::string & collName ) const
{
  // Move to wild-card parser, if needed only
  if ( collName.find( wildcard_ ) != std::string::npos ) {
    // True, if collection name is simply fine
    if ( hasAnyName( collName, std::vector< std::string >( 1, collection() ) ) ) return true;
    // Check, if collection name possibly fits in an edm::InputTag approach
    const edm::InputTag collectionTag( collection() );
    const edm::InputTag collTag( collName );
    // If evaluated collection tag contains a process name, it must have been found already by identity check
    if ( collTag.process().empty() ) {
      // Check instance ...
      if ( ( collTag.instance().empty() && collectionTag.instance().empty() ) || hasAnyName( collTag.instance(), std::vector< std::string >( 1, collectionTag.instance() ) ) ) {
        // ... and label
        return hasAnyName( collTag.label(), std::vector< std::string >( 1, collectionTag.label() ) );
      }
    }
    return false;
  }
  // Use parent class's method otherwise
  return TriggerObject::hasCollection( collName );
}
