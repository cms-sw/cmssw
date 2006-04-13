#ifndef Services_FpeHandler_h
#define Services_FpeHandler_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     EnableFloatingPointExceptions
// 
/**\class EnableFloatingPointExceptions EnableFloatingPointExceptions.h FWCore/Services/src/EnableFloatingPointExceptions.h

 Description: This service gives cmsRun users the ability to configure FPE behavior.  For now the whole job is configured the same way.  A future inhancement would allow control on a module by module basis.


*/
//
// Original Author:  E. Sexton-Kennedy
//         Created:  Tue Apr 11 13:43:16 CDT 2006
// $Id: EnableFloatingPointExceptions.h,v 1.3 2006/03/05 16:42:27 chrjones Exp $
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
