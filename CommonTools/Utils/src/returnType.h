#ifndef CommonTools_Utils_returnType_h
#define CommonTools_Utils_returnType_h
/* \function returnType
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: returnType.h,v 1.3 2012/08/03 18:08:11 wmtan Exp $
 */

#include "CommonTools/Utils/src/TypeCode.h"
#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace reco {
  edm::TypeWithDict returnType(const edm::FunctionWithDict &);
  method::TypeCode returnTypeCode(const edm::FunctionWithDict &);
  method::TypeCode typeCode(const edm::TypeWithDict &);
}

#endif
