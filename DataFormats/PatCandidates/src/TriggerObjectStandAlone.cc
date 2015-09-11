//
//

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include <boost/algorithm/string.hpp>
#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace pat;


// Const data members' definitions


const char TriggerObjectStandAlone::wildcard_;


// Constructors and Destructor


// Default constructor
TriggerObjectStandAlone::TriggerObjectStandAlone() :
  TriggerObject()
{
  filterLabels_.clear();
  pathNames_.clear();
  pathLastFilterAccepted_.clear();
  pathL3FilterAccepted_.clear();
}


// Constructor from pat::TriggerObject
TriggerObjectStandAlone::TriggerObjectStandAlone( const TriggerObject & trigObj ) :
  TriggerObject( trigObj )
{
  filterLabels_.clear();
  pathNames_.clear();
  pathLastFilterAccepted_.clear();
  pathL3FilterAccepted_.clear();
}


// Constructor from trigger::TriggerObject
TriggerObjectStandAlone::TriggerObjectStandAlone( const trigger::TriggerObject & trigObj ) :
  TriggerObject( trigObj )
{
  filterLabels_.clear();
  pathNames_.clear();
  pathLastFilterAccepted_.clear();
  pathL3FilterAccepted_.clear();
}


// Constructor from reco::Candidate
TriggerObjectStandAlone::TriggerObjectStandAlone( const reco::LeafCandidate & leafCand ) :
  TriggerObject( leafCand )
{
  filterLabels_.clear();
  pathNames_.clear();
  pathLastFilterAccepted_.clear();
  pathL3FilterAccepted_.clear();
}


// Constructors from Lorentz-vectors and (optional) PDG ID
TriggerObjectStandAlone::TriggerObjectStandAlone( const reco::Particle::LorentzVector & vec, int id ) :
  TriggerObject( vec, id )
{
  filterLabels_.clear();
  pathNames_.clear();
  pathLastFilterAccepted_.clear();
  pathL3FilterAccepted_.clear();
}
TriggerObjectStandAlone::TriggerObjectStandAlone( const reco::Particle::PolarLorentzVector & vec, int id ) :
  TriggerObject( vec, id )
{
  filterLabels_.clear();
  pathNames_.clear();
  pathLastFilterAccepted_.clear();
  pathL3FilterAccepted_.clear();
}


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
void TriggerObjectStandAlone::addPathOrAlgorithm( const std::string & name, bool pathLastFilterAccepted, bool pathL3FilterAccepted )
{
  checkIfPathsAreUnpacked();

  // Check, if path is already assigned
  if ( ! hasPathOrAlgorithm( name, false, false ) ) {
    // The path itself
    pathNames_.push_back( name );
    // The corresponding usage of the trigger objects
    pathLastFilterAccepted_.push_back( pathLastFilterAccepted );
    pathL3FilterAccepted_.push_back( pathL3FilterAccepted );
  // Enable status updates
  } else if ( pathLastFilterAccepted || pathL3FilterAccepted ) {
    // Search for path
    unsigned index( 0 );
    while ( index < pathNames_.size() ) {
      if ( pathNames_.at( index ) == name ) break;
      ++index;
    }
    // Status update
    if ( index < pathNames_.size() ) {
      pathLastFilterAccepted_.at( index ) = pathLastFilterAccepted_.at( index ) || pathLastFilterAccepted;
      pathL3FilterAccepted_.at( index )   = pathL3FilterAccepted_.at( index )   || pathL3FilterAccepted;
    }
  }
}


// Gets all HLT path or L1 algorithm names
std::vector< std::string > TriggerObjectStandAlone::pathsOrAlgorithms( bool pathLastFilterAccepted, bool pathL3FilterAccepted ) const
{
  checkIfPathsAreUnpacked();

  // Deal with older PAT-tuples, where trigger object usage is not available
  if ( ! hasLastFilter() ) pathLastFilterAccepted = false;
  if ( ! hasL3Filter() ) pathL3FilterAccepted = false;
  // All path names, if usage not restricted (not required or not available)
  if ( ! pathLastFilterAccepted && ! pathL3FilterAccepted ) return pathNames_;
  // Temp vector of path names
  std::vector< std::string > paths;
  // Loop over usage vector and fill corresponding paths into temp vector
  for ( unsigned iPath = 0; iPath < pathNames_.size(); ++iPath ) {
    if ( ( ! pathLastFilterAccepted || pathLastFilterAccepted_.at( iPath ) ) && ( ! pathL3FilterAccepted || pathL3FilterAccepted_.at( iPath ) ) ) paths.push_back( pathNames_.at( iPath ) ); // order matters in order to protect from empty vectors in old data
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
bool TriggerObjectStandAlone::hasPathOrAlgorithm( const std::string & name, bool pathLastFilterAccepted, bool pathL3FilterAccepted ) const
{
  checkIfPathsAreUnpacked();

  // Move to wild-card parser, if needed
  if ( name.find( wildcard_ ) != std::string::npos ) return hasAnyName( name, pathsOrAlgorithms( pathLastFilterAccepted, pathL3FilterAccepted ) );
  // Deal with older PAT-tuples, where trigger object usage is not available
  if ( ! hasLastFilter() ) pathLastFilterAccepted = false;
  if ( ! hasL3Filter() ) pathL3FilterAccepted = false;
  // Check, if path name is assigned at all
  std::vector< std::string >::const_iterator match( std::find( pathNames_.begin(), pathNames_.end(), name ) );
  // False, if path name not assigned
  if ( match == pathNames_.end() ) return false;
  if ( ! pathLastFilterAccepted && ! pathL3FilterAccepted ) return true;
  bool foundLastFilter( pathLastFilterAccepted ? pathLastFilterAccepted_.at( match - pathNames_.begin() ) : true );
  bool foundL3Filter( pathL3FilterAccepted ? pathL3FilterAccepted_.at( match - pathNames_.begin() ) : true );
  // Return for assigned path name, if trigger object usage meets requirement
  return ( foundLastFilter && foundL3Filter );
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


bool TriggerObjectStandAlone::checkIfPathsAreUnpacked(bool throwIfPacked) const {
   bool unpacked = (!pathNames_.empty() || pathIndices_.empty());
   if (!unpacked && throwIfPacked) throw cms::Exception("RuntimeError", "This TriggerObjectStandAlone object has packed trigger path names. Before accessing path names you must call unpackPathNames with an edm::TriggerNames object. You can get the latter from the edm::Event or fwlite::Event and the TriggerResults\n");
   return unpacked;
}

void TriggerObjectStandAlone::packPathNames(const edm::TriggerNames &names) {
    if (!pathIndices_.empty()) {
        if (!pathNames_.empty()) {
            throw cms::Exception("RuntimeError", "Error, trying to pack a partially packed TriggerObjectStandAlone");
        } else {
            return;
        }
    }
    bool ok = true;
    unsigned int n = pathNames_.size(), end = names.size();
    std::vector<uint16_t> indices(n); 
    for (unsigned int i = 0; i < n; ++i) {
        uint16_t id = names.triggerIndex(pathNames_[i]);
        if (id >= end) {
            static std::atomic<int> _warn(0);
            if (++_warn < 5) edm::LogWarning("TriggerObjectStandAlone::packPathNames()") << "Warning: can't resolve '" << pathNames_[i] << "' to a path index" << std::endl;
            ok = false; break;
        } else {
            indices[i] = id;
        }
    }
    if (ok) {
        pathIndices_.swap(indices);
        pathNames_.clear();
    }
}

void TriggerObjectStandAlone::unpackPathNames(const edm::TriggerNames &names) {
    if (!pathNames_.empty()) {
        if (!pathIndices_.empty()) {
            throw cms::Exception("RuntimeError", "Error, trying to unpack a partially unpacked TriggerObjectStandAlone");
        } else {
            return;
        }
    }
    unsigned int n = pathIndices_.size(), end = names.size();
    std::vector<std::string> paths(n); 
    for (unsigned int i = 0; i < n; ++i) {
        if (pathIndices_[i] >= end) throw cms::Exception("RuntimeError", "Error, path index out of bounds");
        paths[i] = names.triggerName(pathIndices_[i]);
    }
    pathIndices_.clear();
    pathNames_.swap(paths);
}

