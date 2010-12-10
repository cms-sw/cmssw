// $Id: EventStreamConfigurationInfo.cc,v 1.8 2010/08/06 20:24:31 wmtan Exp $
/// @file: EventStreamConfigurationInfo.cc

#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include <ostream>

using stor::EventStreamConfigurationInfo;
using namespace std;

ostream&
stor::operator << ( ostream& os,
                    const EventStreamConfigurationInfo& ci )
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
