#ifndef Utilities_OutletBase_h
#define Utilities_OutletBase_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     OutletBase
// 
/**\class OutletBase OutletBase.h FWCore/Utilities/interface/OutletBase.h

 Description: Base class for 'outlets' which can be used to send data through edm::ExtensionCords

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep 22 12:39:16 EDT 2006
//

// system include files

// user include files
#include "FWCore/Utilities/interface/ExtensionCord.h"

// forward declarations
namespace edm {
template <class T>
class OutletBase
{

   protected:
      OutletBase( ExtensionCord<T>& iCord) : cord_(iCord) {}
      virtual ~OutletBase() { this->setGetter(0); }

      void setGetter( extensioncord::ECGetterBase<T>* iGetter) {
        cord_.setGetter(iGetter);
      }


  
   private:
      OutletBase(const OutletBase&); // stop default

      const OutletBase& operator=(const OutletBase&); // stop default

      // ---------- member data --------------------------------
      ExtensionCord<T>& cord_;
};
}

#endif
