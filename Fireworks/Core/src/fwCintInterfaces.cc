// -*- C++ -*-
//
// Package:     Core
// Class  :     fwCintInterfaces
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Jun 12 15:00:52 EDT 2008
// $Id: fwCintInterfaces.cc,v 1.2 2008/11/06 22:05:27 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/src/fwCintInterfaces.h"


void fwSetInCint(double iValue)
{
   fwCintReturnType()=kFWCintReturnDouble;
   fwGetFromCintDouble() = iValue;
}

void fwSetInCint(long iValue)
{
   fwCintReturnType()=kFWCintReturnLong;
   fwGetFromCintLong() = iValue;
}

static void* s_objectPtr=0;
void* fwGetObjectPtr()
{
   return s_objectPtr;
}

void fwSetObjectPtr(const void* iValue)
{
   s_objectPtr=const_cast<void*>(iValue);
}

FWCintReturnTypes& fwCintReturnType() {
   static FWCintReturnTypes s_type=kFWCintReturnNoReturn;
   return s_type;
}

double& fwGetFromCintDouble()
{
   static double s_value = 0;
   return s_value;
}

long& fwGetFromCintLong()
{
   static long s_value = 0;
   return s_value;
}
