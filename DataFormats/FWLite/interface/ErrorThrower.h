#ifndef DataFormats_FWLite_ErrorThrower_h
#define DataFormats_FWLite_ErrorThrower_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     ErrorThrower
// 
/**\class ErrorThrower ErrorThrower.h DataFormats/FWLite/interface/ErrorThrower.h

 Description: Allows delaying a throw of a specific exception

 Usage:
    Used internally by FWLite

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Sep 23 09:58:07 EDT 2008
//

#if !defined(__CINT__) && !defined(__MAKECINT__)
// system include files
#include <typeinfo>

// user include files

// forward declarations
namespace  fwlite {
   class ErrorThrower {

   public:
      ErrorThrower();
      virtual ~ErrorThrower();

      // ---------- const member functions ---------------------
      virtual void throwIt() const =0;
      virtual ErrorThrower* clone() const =0;
      
      // ---------- static member functions --------------------
      static ErrorThrower* unsetErrorThrower();
      static ErrorThrower* errorThrowerBranchNotFoundException(const std::type_info&, const char*, const char*, const char*);
      static ErrorThrower* errorThrowerProductNotFoundException(const std::type_info&, const char*, const char*, const char*);
      
      // ---------- member functions ---------------------------
      
   private:
      //ErrorThrower(const ErrorThrower&); // stop default

      //const ErrorThrower& operator=(const ErrorThrower&); // stop default

      // ---------- member data --------------------------------

   };

}
#endif
#endif
