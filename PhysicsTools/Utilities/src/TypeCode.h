#ifndef PhysicsTools_Utilites_TypeCode_h
#define PhysicsTools_Utilites_TypeCode_h
/* \class TypeCode
 *
 * \author Luca Lista, INFN
 *
 * \version $Id$
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
