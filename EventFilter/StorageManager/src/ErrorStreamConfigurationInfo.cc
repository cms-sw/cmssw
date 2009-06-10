// $Id$

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"

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
