#ifndef CommonTools_Utils_TypeCode_h
#define CommonTools_Utils_TypeCode_h
/* \class TypeCode
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: TypeCode.h,v 1.1 2007/12/20 10:07:15 llista Exp $
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
      boolType,
      invalid
    };
  }
}

#endif
