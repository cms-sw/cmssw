// $Id: TransitionRecord.cc,v 1.2 2009/06/10 08:15:28 dshpakov Exp $
/// @file: TransitionRecord.cc

#include "EventFilter/StorageManager/interface/TransitionRecord.h"

#include <iostream>
#include <sys/time.h>

using namespace stor;
using namespace std;

TransitionRecord::TransitionRecord( const std::string& state_name,
				    bool is_entry ):
  _stateName( state_name ),
  _isEntry( is_entry )
{
  gettimeofday( &_timestamp, NULL );
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
