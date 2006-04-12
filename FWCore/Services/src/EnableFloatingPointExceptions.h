#ifndef Services_EnableFloatingPointExceptions_h
#define Services_EnableFloatingPointExceptions_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     EnableFloatingPointExceptions
// 
/**\class EnableFloatingPointExceptions EnableFloatingPointExceptions.h FWCore/Services/interface/EnableFloatingPointExceptions.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Apr 12 09:27:25 EDT 2006
// $Id$
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
namespace edm {
  namespace service {
class EnableFloatingPointExceptions
{

   public:
      EnableFloatingPointExceptions(const ParameterSet&);
      virtual ~EnableFloatingPointExceptions();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      EnableFloatingPointExceptions(const EnableFloatingPointExceptions&); // stop default

      const EnableFloatingPointExceptions& operator=(const EnableFloatingPointExceptions&); // stop default

      void enable(bool) const;
      // ---------- member data --------------------------------
      int initialMask_;
};

  }
}

#endif
