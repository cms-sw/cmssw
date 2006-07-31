#ifndef Services_FpeHandler_h
#define Services_FpeHandler_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     EnableFloatingPointExceptions
// 
/**\class EnableFloatingPointExceptions EnableFloatingPointExceptions.h FWCore/Services/src/EnableFloatingPointExceptions.h

 Description: This service gives cmsRun users the ability to configure FPE behavior.  For now the whole job is configured the same way.  A future inhancement would allow control on a module by module basis.

 Usage:
    By adding the following to your configuration
    /code
      service = EnableFloatingPointExceptions {}
    /endcode
   Any arithmetic 'exception' (such as divide by zero) will cause the CPU to emit a SIGFPE.  This is very useful if
you are trying to track down where a floating point value of 'nan' or 'inf' is being generated.
*/
//
// Original Author:  E. Sexton-Kennedy
//         Created:  Tue Apr 11 13:43:16 CDT 2006
// $Id: EnableFloatingPointExceptions.h,v 1.2 2006/04/13 14:29:43 lsexton Exp $
//

// system include files

// user include files

// forward declarations

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"


namespace edm {
   namespace service {
      class EnableFloatingPointExceptions {
public:
         EnableFloatingPointExceptions(const ParameterSet&,ActivityRegistry&);
         
         void preModule(const ModuleDescription&);
         void postModule(const ModuleDescription&);
private:
	 void controlFpe();
         bool enableDivByZeroEx_;
         bool enableInvalidEx_;
         bool enableOverFlowEx_;
	 bool enableUnderFlowEx_;
      };
   }
}



#endif
