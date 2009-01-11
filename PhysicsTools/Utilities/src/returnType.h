#ifndef PhysicsTools_Utilities_returnType_h
#define PhysicsTools_Utilities_returnType_h
/* \function returnType
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: returnType.h,v 1.2 2007/12/20 15:47:49 llista Exp $
 */

#include "PhysicsTools/Utilities/src/TypeCode.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"

namespace reco {
  Reflex::Type returnType(const Reflex::Member &);
  method::TypeCode returnTypeCode(const Reflex::Member &);
  method::TypeCode typeCode(const Reflex::Type &);
}

#endif
