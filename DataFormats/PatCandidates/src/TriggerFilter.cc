//
// $Id: TriggerFilter.cc,v 1.1.2.8 2009/02/20 13:47:41 vadler Exp $
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
  for ( std::vector< unsigned >::const_iterator iO = objectKeys_.begin(); iO != objectKeys_.end(); ++iO ) {
    if ( *iO == objectKey ) {
      return true;
    }
  }
  return false;
}

bool TriggerFilter::hasObjectId( unsigned objectId ) const
{
  for ( std::vector< unsigned >::const_iterator iO = objectIds_.begin(); iO != objectIds_.end(); ++iO ) {
    if ( *iO == objectId ) {
      return true;
    }
  }
  return false;
}
