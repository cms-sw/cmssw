#ifndef PhysicsTools_Utilities_returnType_h
#define PhysicsTools_Utilities_returnType_h
/* \function returnType
 *
 * \author Luca Lista, INFN
 *
 * \version $Id$
 */

#include "PhysicsTools/Utilities/src/TypeCode.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"

namespace reco {
  method::TypeCode returnTypeCode(const ROOT::Reflex::Member &);
  ROOT::Reflex::Type returnType(const ROOT::Reflex::Member &);
}

#endif
