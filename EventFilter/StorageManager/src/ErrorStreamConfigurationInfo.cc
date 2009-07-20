// $Id: ErrorStreamConfigurationInfo.cc,v 1.2 2009/06/10 08:15:26 dshpakov Exp $
/// @file: ErrorStreamConfigurationInfo.cc

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
