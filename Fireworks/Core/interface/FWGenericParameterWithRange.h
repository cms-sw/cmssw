#ifndef Fireworks_Core_FWGenericParameterWithRange_h
#define Fireworks_Core_FWGenericParameterWithRange_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGenericParameterWithRange
//
/**\class FWGenericParameterWithRange FWGenericParameter.h Fireworks/Core/interface/FWLongParameter.h

   Description: Provides access to a simple double parameter

   Usage:
    If min and max values are both identical than no restriction is placed on the allowed value

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Mar  7 14:36:34 EST 2008
// $Id: FWGenericParameterWithRange.h,v 1.4 2012/02/22 03:45:57 amraktad Exp $
//

// user include files
#include "Fireworks/Core/interface/FWGenericParameter.h"

// forward declarations

template<class T>
class FWGenericParameterWithRange : public FWGenericParameter<T>
{
public:
   FWGenericParameterWithRange(void) :
      FWGenericParameter<T>(),
      m_min(-1),
      m_max(-1)
   {}

   FWGenericParameterWithRange(FWParameterizable* iParent,
                               const std::string& iName,
                               const T &iDefault=T(),
                               T iMin=-1,
                               T iMax=-1) :
      FWGenericParameter<T>(iParent, iName, iDefault),
      m_min(iMin),
      m_max(iMax)
   {}

   template <class K>
   FWGenericParameterWithRange(FWParameterizable* iParent,
                               const std::string& iName,
                               K iCallback,
                               const T &iDefault=T(),
                               T iMin=-1,
                               T iMax=-1) :
      FWGenericParameter<T>(iParent, iName, iCallback, iDefault),
      m_min(iMin),
      m_max(iMax)
   {}

   // ---------- const member functions ---------------------

   T min() const { return m_min; }
   T max() const { return m_max; }

private:

   T m_min;
   T m_max;
};

#endif
