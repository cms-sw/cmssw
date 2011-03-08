//
// $Id: TriggerFilter.cc,v 1.3 2009/04/27 20:45:18 vadler Exp $
//


#include "DataFormats/PatCandidates/interface/TriggerFilter.h"


using namespace pat;


/// default constructor

TriggerFilter::TriggerFilter() :
  label_(),
  type_(),
  status_()
{
  objectIds_.clear();
}

/// constructor from values

TriggerFilter::TriggerFilter( const std::string & label, int status ) :
  label_( label ),
  type_(),
  status_( status )
{
  objectIds_.clear();
}

TriggerFilter::TriggerFilter( const edm::InputTag & tag, int status ) :
  label_( tag.label() ),
  type_(),
  status_( status )
{
  objectIds_.clear();
}

/// setters

// only -1,0,1 accepted; returns 'false' (and does not modify the status) otherwise
bool TriggerFilter::setStatus( int status )
{
  if ( status < -1 || 1 < status ) {
    return false;
  }
  status_ = status;
  return true;
}

/// getters

bool TriggerFilter::hasObjectKey( unsigned objectKey ) const
{
  for ( size_t iO = 0; iO < objectKeys().size(); ++iO ) {
    if ( objectKeys().at( iO ) == objectKey ) {
      return true;
    }
  }
  return false;
}

bool TriggerFilter::hasObjectId( int objectId ) const
{
  for ( size_t iO = 0; iO < objectIds().size(); ++iO ) {
    if ( objectIds().at( iO ) == objectId ) {
      return true;
    }
  }
  return false;
}
