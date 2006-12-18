#ifndef Services_FpeHandler_h
#define Services_FpeHandler_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     EnableFloatingPointExceptions
// 
/**\class EnableFloatingPointExceptions EnableFloatingPointExceptions.h FWCore/Services/src/EnableFloatingPointExceptions.h

    Description: This service gives cmsRun users the ability to
    configure the behavior of the Floating Point (FP) Processor.
    For now the whole job is configured the same way.  A future
    enhancement might allow control on a module by module basis.

    Usage:  There are two separate aspects of the FP environment
    this service can control: 
        1. exceptions
        2. precision control on x87 FP processors.

    If you do not use the service at all, floating point exceptions
    will not be trapped (FP exceptions will not cause a crash).  Add
    the following to the configuration file to enable to the exceptions:

      service = EnableFloatingPointExceptions 
      {
          untracked bool enableDivByZeroEx = true
          untracked bool enableInvalidEx = true
          untracked bool enableOverFlowEx = true
          untracked bool enableUnderFlowEx = true
      }

    The defaults for these options are currently all false.
    (in an earlier version DivByZero, Invalid, and Overflow
    defaulted to true, we hope to return to those defaults
    someday when the frequency of such exceptions has decreased)

    Enabling exceptions is very useful if you are trying to
    track down where a floating point value of 'nan' or 'inf'
    is being generated and is even better if the goal is to
    eliminate them.

    One can also control the precision of floating point
    operations in x87 FP processor.

      service = EnableFloatingPointExceptions 
      {
          untracked bool setPrecisionDouble = true
      }

    If set true (the default if the service is used), the
    floating precision in the x87 math processor will be
    set to round results of addition, subtraction, multiplication,
    division, and square root to 64 bits after each operation
    instead of the x87 default, which is 80 bits for values
    in registers (this is the default you get if this
    service is not used at all).

    The precision control only affects Intel and AMD 32 bit CPUs
    under LINUX.  We have not implemented precision control in the
    service for other CPUs (some other CPUs round to 64 bits by
    default and often CPUs do not allow control of the precision
    of floating point calculations).
*/
//
// Original Author:  E. Sexton-Kennedy
//         Created:  Tue Apr 11 13:43:16 CDT 2006
// $Id: EnableFloatingPointExceptions.h,v 1.3 2006/07/31 12:57:59 chrjones Exp $
//

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

         bool setPrecisionDouble_;
      };
   }
}

#endif
