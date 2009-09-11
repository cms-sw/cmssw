// $Id: ErrorStreamConfigurationInfo.cc,v 1.3 2009/07/20 13:07:27 mommsen Exp $
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
