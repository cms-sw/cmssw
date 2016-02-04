// $Id: ErrorStreamConfigurationInfo.cc,v 1.6 2010/12/16 16:35:29 mommsen Exp $
/// @file: ErrorStreamConfigurationInfo.cc

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include <ostream>

using stor::ErrorStreamConfigurationInfo;


bool ErrorStreamConfigurationInfo::operator<(const ErrorStreamConfigurationInfo& other) const
{
  if ( _streamLabel != other.streamLabel() )
    return ( _streamLabel < other.streamLabel() );
  if ( _streamId != other.streamId() )
    return ( _streamId < other.streamId() );
  return ( _maxFileSizeMB < other.maxFileSizeMB() );
}

std::ostream& stor::operator<<
(
    std::ostream& os,
    const ErrorStreamConfigurationInfo& ci
)
{
  os << "ErrorStreamConfigurationInfo:" << std::endl
     << " Stream label: " << ci.streamLabel() << std::endl
     << " Maximum file size, MB: " << ci.maxFileSizeMB() << std::endl
     << " Stream Id: " << ci.streamId() << std::endl;

  return os;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
