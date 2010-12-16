// $Id: EventStreamConfigurationInfo.cc,v 1.9 2010/12/10 13:23:43 mommsen Exp $
/// @file: EventStreamConfigurationInfo.cc

#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include <ostream>

using stor::EventStreamConfigurationInfo;


bool EventStreamConfigurationInfo::operator<(const EventStreamConfigurationInfo& other) const
{
  if ( _outputModuleLabel != other.outputModuleLabel() )
    return ( _outputModuleLabel < other.outputModuleLabel() );
  if ( _triggerSelection != other.triggerSelection() )
    return ( _triggerSelection < other.triggerSelection() );
  if ( _selEvents != other.selEvents() )
    return ( _selEvents < other.selEvents() );
  if ( _streamLabel != other.streamLabel() )
    return ( _streamLabel < other.streamLabel() );
  if ( _streamId != other.streamId() )
    return ( _streamId < other.streamId() );
  if ( _maxFileSizeMB != other.maxFileSizeMB() )
    return ( _maxFileSizeMB < other.maxFileSizeMB() );
  return ( _fractionToDisk < other.fractionToDisk() );
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
    for( unsigned int i = 0; i < ci.selEvents().size(); ++i )
    {
      os << std::endl << "  " << ci.selEvents()[i];
    }
  
  return os;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
