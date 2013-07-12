// $Id: DiskWriter.cc,v 1.5 2009/07/10 14:51:12 dshpakov Exp $
/// @file: EnquingPolicyTag.cc

#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"

using namespace stor;

std::ostream& enquing_policy::operator << ( std::ostream& os,
					    const enquing_policy::PolicyTag& ptag )
{
  switch( ptag )
    {
    case enquing_policy::DiscardNew:
      os << "Discard New";
      break;
    case enquing_policy::DiscardOld:
      os << "Discard Old";
      break;
    case enquing_policy::FailIfFull:
      os << "Fail If Full";
      break;
    case enquing_policy::Max:
      os << "Max";
      break;
    default:
      os << "BUG: Undefined Policy";
    }
  return os;
}
