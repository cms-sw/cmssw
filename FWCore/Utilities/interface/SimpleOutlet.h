#ifndef Utilities_SimpleOutlet_h
#define Utilities_SimpleOutlet_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     SimpleOutlet
// 
/**\class SimpleOutlet SimpleOutlet.h FWCore/Utilities/interface/SimpleOutlet.h

 Description: A simple outlet that works with the edm::ExtensionCord

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep 22 13:10:36 EDT 2006
// $Id$
//

// system include files

// user include files
#include "FWCore/Utilities/interface/OutletBase.h"
#include "FWCore/Utilities/interface/ECGetterBase.h"

// forward declarations
namespace edm {
  template<class T>
  class SimpleOutlet : private OutletBase<T>
  {
    
    public:
      SimpleOutlet( extensioncord::ECGetterBase<T>& iGetter,
                    ExtensionCord<T>& iCord) :
        OutletBase<T>(iCord) {
          this->setGetter(&iGetter);
      }

   private:
      SimpleOutlet(const SimpleOutlet&); // stop default

      const SimpleOutlet& operator=(const SimpleOutlet&); // stop default

};

}
#endif
