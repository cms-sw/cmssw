//
// $Id: TriggerFilter.cc,v 1.7 2011/05/24 15:56:25 vadler Exp $
//


#include "DataFormats/PatCandidates/interface/TriggerFilter.h"


using namespace pat;


// Constructors and Destructor


// Default constructor
TriggerFilter::TriggerFilter() :
  label_(),
  type_(),
  status_(),
  saveTags_()
{
  objectKeys_.clear();
  triggerObjectTypes_.clear();
}


// Constructor from std::string for filter label
TriggerFilter::TriggerFilter( const std::string & label, int status, bool saveTags ) :
  label_( label ),
  type_(),
  status_( status ),
  saveTags_( saveTags )
{
  objectKeys_.clear();
  triggerObjectTypes_.clear();
}


// Constructor from edm::InputTag for filter label
TriggerFilter::TriggerFilter( const edm::InputTag & tag, int status, bool saveTags ) :
  label_( tag.label() ),
  type_(),
  status_( status ),
  saveTags_( saveTags )
{
  objectKeys_.clear();
  triggerObjectTypes_.clear();
}


// Methods


// Set the filter status
bool TriggerFilter::setStatus( int status )
{
  if ( status < -1 || 1 < status ) return false;
  status_ = status;
  return true;
}


// Get all trigger object type identifiers
std::vector< int > TriggerFilter::triggerObjectTypes() const
{
  std::vector< int > triggerObjectTypes;
  for ( size_t iTo = 0; iTo < triggerObjectTypes_.size(); ++iTo ) {
    triggerObjectTypes.push_back( triggerObjectTypes_.at( iTo ) );
  }
  return triggerObjectTypes;
}


// Checks, if a certain trigger object collection index is assigned
bool TriggerFilter::hasObjectKey( unsigned objectKey ) const
{
  for ( size_t iO = 0; iO < objectKeys().size(); ++iO ) {
    if ( objectKeys().at( iO ) == objectKey ) return true;
  }
  return false;
}


// Checks, if a certain trigger object type identifier is assigned
bool TriggerFilter::hasTriggerObjectType( trigger::TriggerObjectType triggerObjectType ) const
{
  for ( size_t iO = 0; iO < triggerObjectTypes().size(); ++iO ) {
    if ( triggerObjectTypes().at( iO ) == triggerObjectType ) return true;
  }
  return false;
}
