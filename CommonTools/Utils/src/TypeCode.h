#ifndef CommonTools_Utils_TypeCode_h
#define CommonTools_Utils_TypeCode_h
/* \class TypeCode
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: TypeCode.h,v 1.2 2009/12/04 13:29:43 gpetrucc Exp $
 *
 */

namespace reco {
  namespace method {
    enum TypeCode { 
      doubleType = 0, floatType,
      intType, uIntType,
      charType, uCharType,
      shortType, uShortType, 
      longType, uLongType, 
      boolType, enumType,
      invalid
    };
  }
}

#endif
