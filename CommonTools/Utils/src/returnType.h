#ifndef CommonTools_Utils_returnType_h
#define CommonTools_Utils_returnType_h
/* \function returnType
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: returnType.h,v 1.2 2012/06/26 21:13:13 wmtan Exp $
 */

#include "CommonTools/Utils/src/TypeCode.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace reco {
  edm::TypeWithDict returnType(const edm::MemberWithDict &);
  method::TypeCode returnTypeCode(const edm::MemberWithDict &);
  method::TypeCode typeCode(const edm::TypeWithDict &);
}

#endif
