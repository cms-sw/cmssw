/*----------------------------------------------------------------------
  
$Id: TypeID.cc,v 1.2 2005/04/06 12:27:10 paterno Exp $

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
  operator<< (std::ostream& os, const TypeID& id)
  {
    id.print(os);
    return os;
  }

  
}
