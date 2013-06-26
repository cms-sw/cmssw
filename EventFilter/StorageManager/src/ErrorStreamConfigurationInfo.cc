// $Id: ErrorStreamConfigurationInfo.cc,v 1.7 2011/03/07 15:31:32 mommsen Exp $
/// @file: ErrorStreamConfigurationInfo.cc

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include <ostream>

using stor::ErrorStreamConfigurationInfo;


bool ErrorStreamConfigurationInfo::operator<(const ErrorStreamConfigurationInfo& other) const
{
  if ( streamLabel_ != other.streamLabel() )
    return ( streamLabel_ < other.streamLabel() );
  if ( streamId_ != other.streamId() )
    return ( streamId_ < other.streamId() );
  return ( maxFileSizeMB_ < other.maxFileSizeMB() );
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
