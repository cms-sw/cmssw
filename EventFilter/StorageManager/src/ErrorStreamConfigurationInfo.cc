// $Id: ErrorStreamConfigurationInfo.cc,v 1.4 2009/09/11 21:07:06 elmer Exp $
/// @file: ErrorStreamConfigurationInfo.cc

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include <ostream>

using stor::ErrorStreamConfigurationInfo;
using namespace std;

ostream&
stor::operator << ( ostream& os,
		    const ErrorStreamConfigurationInfo& ci )
{

  os << "ErrorStreamConfigurationInfo:" << endl
     << " Stream label: " << ci.streamLabel() << endl
     << " Maximum file size, MB: " << ci.maxFileSizeMB() << endl
     << " Stream Id: " << ci.streamId() << endl;

  return os;

}
