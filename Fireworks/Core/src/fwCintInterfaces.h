#ifndef Fireworks_Core_fwCintInterfaces_h
#define Fireworks_Core_fwCintInterfaces_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     fwCintInterfaces
//
/**\class fwCintInterfaces fwCintInterfaces.h Fireworks/Core/interface/fwCintInterfaces.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Jun 12 15:00:50 EDT 2008
// $Id: fwCintInterfaces.h,v 1.3 2009/01/23 21:35:44 amraktad Exp $
//

// system include files

// user include files

// forward declarations

void fwSetInCint(double);
void fwSetInCint(long);

void* fwGetObjectPtr();

#if !defined(__CINT__) && !defined(__MAKECINT__)
void fwSetObjectPtr(const void*);
enum FWCintReturnTypes
{
   kFWCintReturnNoReturn,
   kFWCintReturnDouble,
   kFWCintReturnLong
};

FWCintReturnTypes& fwCintReturnType();
double& fwGetFromCintDouble();
long& fwGetFromCintLong();

#endif
#endif
