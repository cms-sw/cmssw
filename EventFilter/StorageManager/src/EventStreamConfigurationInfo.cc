// $Id: EventStreamConfigurationInfo.cc,v 1.12 2011/03/07 15:31:32 mommsen Exp $
/// @file: EventStreamConfigurationInfo.cc

#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include <ostream>

using stor::EventStreamConfigurationInfo;


bool EventStreamConfigurationInfo::operator<(const EventStreamConfigurationInfo& other) const
{
  if ( outputModuleLabel_ != other.outputModuleLabel() )
    return ( outputModuleLabel_ < other.outputModuleLabel() );
  if ( triggerSelection_ != other.triggerSelection() )
    return ( triggerSelection_ < other.triggerSelection() );
  if ( eventSelection_ != other.eventSelection() )
    return ( eventSelection_ < other.eventSelection() );
  if ( streamLabel_ != other.streamLabel() )
    return ( streamLabel_ < other.streamLabel() );
  if ( streamId_ != other.streamId() )
    return ( streamId_ < other.streamId() );
  if ( maxFileSizeMB_ != other.maxFileSizeMB() )
    return ( maxFileSizeMB_ < other.maxFileSizeMB() );
  return ( fractionToDisk_ < other.fractionToDisk() );
}


std::ostream& stor::operator<<
(
  std::ostream& os,
  const EventStreamConfigurationInfo& ci
)
{
  os << "EventStreamConfigurationInfo:" << std::endl
     << " Stream label: " << ci.streamLabel() << std::endl
     << " Maximum file size, MB: " << ci.maxFileSizeMB() << std::endl
     << " HLT output: " << ci.outputModuleLabel() << std::endl
     << " Fraction of events written to disk: " << ci.fractionToDisk() << std::endl
     << " Stream Id: " << ci.streamId() << std::endl;
  
  os << " Event filters:";
  if (ci.triggerSelection().size()) {
    os << std::endl << ci.triggerSelection();
  }
  else
    for( unsigned int i = 0; i < ci.eventSelection().size(); ++i )
    {
      os << std::endl << "  " << ci.eventSelection()[i];
    }
  
  return os;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
