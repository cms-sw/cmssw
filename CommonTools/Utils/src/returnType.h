#ifndef CommonTools_Utils_returnType_h
#define CommonTools_Utils_returnType_h
/* \function returnType
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: returnType.h,v 1.3 2009/01/11 23:37:33 hegner Exp $
 */

#include "CommonTools/Utils/src/TypeCode.h"
#include "Reflex/Member.h"
#include "Reflex/Type.h"

namespace reco {
  Reflex::Type returnType(const Reflex::Member &);
  method::TypeCode returnTypeCode(const Reflex::Member &);
  method::TypeCode typeCode(const Reflex::Type &);
}

#endif
