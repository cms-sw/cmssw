//
// $Id: TriggerCondition.cc,v 1.2 2011/11/30 13:39:45 vadler Exp $
//


#include "DataFormats/PatCandidates/interface/TriggerCondition.h"


using namespace pat;


// Constructors and Destructor


// Default constructor
TriggerCondition::TriggerCondition() :
  name_()
, accept_()
, category_()
, type_()
{
  triggerObjectTypes_.clear();
  objectKeys_.clear();
}


// Constructor from condition name "only"
TriggerCondition::TriggerCondition( const std::string & name ) :
  name_( name )
, accept_()
, category_()
, type_()
{
  triggerObjectTypes_.clear();
  objectKeys_.clear();
}


// Constructor from values
TriggerCondition::TriggerCondition( const std::string & name, bool accept ) :
  name_( name )
, accept_( accept )
, category_()
, type_()
{
  triggerObjectTypes_.clear();
  objectKeys_.clear();
}


// Methods


// Get the trigger object types
std::vector< int > TriggerCondition::triggerObjectTypes() const
{
  std::vector< int > triggerObjectTypes;
  for ( size_t iT = 0; iT < triggerObjectTypes_.size(); ++iT ) {
    triggerObjectTypes.push_back( int( triggerObjectTypes_.at( iT ) ) );
  }
  return triggerObjectTypes;
}


// Checks, if a certain trigger object type is assigned
bool TriggerCondition::hasTriggerObjectType( trigger::TriggerObjectType triggerObjectType ) const
{
  for ( size_t iT = 0; iT < triggerObjectTypes_.size(); ++iT ) {
    if ( triggerObjectTypes_.at( iT ) == triggerObjectType ) return true;
  }
  return false;
}


// Checks, if a certain trigger object collection index is assigned
bool TriggerCondition::hasObjectKey( unsigned objectKey ) const
{
  for ( size_t iO = 0; iO < objectKeys_.size(); ++iO ) {
    if ( objectKeys_.at( iO ) == objectKey ) return true;
  }
  return false;
}
