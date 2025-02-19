#ifndef Fireworks_Core_FWParameters_h
#define Fireworks_Core_FWParameters_h
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
// $Id: FWParameters.h,v 1.2 2012/02/22 03:45:57 amraktad Exp $
//

// user include files
#include "Fireworks/Core/interface/FWGenericParameter.h"
#include "Fireworks/Core/interface/FWGenericParameterWithRange.h"

// forward declarations

struct FWParameters
{
  typedef FWGenericParameterWithRange<long>   Long;
  typedef FWGenericParameterWithRange<double> Double;
  typedef FWGenericParameter<std::string>     String;
  typedef FWGenericParameter<bool>            Bool;
};

typedef FWParameters::Long   FWLongParameter;
typedef FWParameters::Double FWDoubleParameter;
typedef FWParameters::String FWStringParameter;
typedef FWParameters::Bool   FWBoolParameter;

#endif
