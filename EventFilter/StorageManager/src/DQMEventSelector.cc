// $Id: DQMEventSelector.cc,v 1.6 2011/03/07 15:31:32 mommsen Exp $
/// @file: DQMEventSelector.cc

#include "EventFilter/StorageManager/interface/DQMEventSelector.h"

using namespace stor;

bool DQMEventSelector::acceptEvent
(
  const I2OChain& ioc,
  const utils::TimePoint_t& now
)
{
  if( registrationInfo_->isStale(now) ) return false;
  if( registrationInfo_->topLevelFolderName() == std::string( "*" ) ) return true;
  if( registrationInfo_->topLevelFolderName() == ioc.topFolderName() ) return true;
  return false;
}


bool DQMEventSelector::operator<(const DQMEventSelector& other) const
{
  if ( queueId() != other.queueId() )
    return ( queueId() < other.queueId() );
  return ( *(registrationInfo_) < *(other.registrationInfo_) );
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
