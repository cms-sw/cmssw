#ifndef PhysicsTools_Utilities_returnType_h
#define PhysicsTools_Utilities_returnType_h
/* \function returnType
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: returnType.h,v 1.1 2007/12/20 10:51:36 llista Exp $
 */

#include "PhysicsTools/Utilities/src/TypeCode.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"

namespace reco {
  ROOT::Reflex::Type returnType(const ROOT::Reflex::Member &);
  method::TypeCode returnTypeCode(const ROOT::Reflex::Member &);
  method::TypeCode typeCode(const ROOT::Reflex::Type &);
}

#endif
