/*----------------------------------------------------------------------
  
$Id: TypeID.cc,v 1.1 2005/05/29 02:29:54 wmtan Exp $

----------------------------------------------------------------------*/
#include <ostream>
#include "FWCore/CoreFramework/src/TypeID.h"

namespace edm
{

  void
  TypeID::print(std::ostream& os) const
  {
    os << t_.name();
  }

  std::ostream&
  operator<<(std::ostream& os, const TypeID& id)
  {
    id.print(os);
    return os;
  }

  
}
