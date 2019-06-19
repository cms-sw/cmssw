#ifndef CommonTools_Utils_returnType_h
#define CommonTools_Utils_returnType_h

/* \function returnType
 *
 * \author Luca Lista, INFN
 *
 */

#include "CommonTools/Utils/src/TypeCode.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"

namespace edm {
  class FunctionWithDict;
}

namespace reco {
  edm::TypeWithDict returnType(const edm::FunctionWithDict&);
  method::TypeCode returnTypeCode(const edm::FunctionWithDict&);
  method::TypeCode typeCode(const edm::TypeWithDict&);
}  // namespace reco

#endif  // CommonTools_Utils_returnType_h
