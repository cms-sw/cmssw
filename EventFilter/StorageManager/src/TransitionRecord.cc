// $Id: TransitionRecord.cc,v 1.5 2011/03/07 15:31:32 mommsen Exp $
/// @file: TransitionRecord.cc

#include "EventFilter/StorageManager/interface/TransitionRecord.h"

#include <ostream>
#include <sys/time.h>

using namespace stor;
using namespace std;

TransitionRecord::TransitionRecord
(
  const std::string& stateName,
  bool isEntry
):
  stateName_( stateName ),
  isEntry_( isEntry )
{
  gettimeofday( &timestamp_, NULL );
}

std::ostream& stor::operator << ( std::ostream& os,
				  const TransitionRecord& tr )
{

  os << tr.timeStamp().tv_sec << "."
     << tr.timeStamp().tv_usec << ": ";

  if( tr.isEntry() )
    {
      os << "entered";
    }
  else
    {
      os << "exited";
    }

  os << " " << tr.stateName();

  return os;

}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
