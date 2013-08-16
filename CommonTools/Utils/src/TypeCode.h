#ifndef CommonTools_Utils_TypeCode_h
#define CommonTools_Utils_TypeCode_h
/* \class TypeCode
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: TypeCode.h,v 1.1 2009/02/24 14:10:22 llista Exp $
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
