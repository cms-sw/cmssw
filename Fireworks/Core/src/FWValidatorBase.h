#ifndef Fireworks_Core_FWValidatorBase_h
#define Fireworks_Core_FWValidatorBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWValidatorBase
//
/**\class FWValidatorBase FWValidatorBase.h Fireworks/Core/interface/FWValidatorBase.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Aug 22 20:42:39 EDT 2008
//

// system include files
#include <vector>
#include <string>
#include <memory>

// user include files

// forward declarations

class FWValidatorBase {

public:
   FWValidatorBase() {
   }
   virtual ~FWValidatorBase() {
   }

   // ---------- const member functions ---------------------
   //fills the vector with
   // first: the full details about the substitution
   // second: exactly what should be inserted into the expression to complete the option
   virtual void fillOptions(const char* iBegin, const char* iEnd,
                            std::vector<std::pair<std::shared_ptr<std::string>, std::string> >& oOptions) const = 0;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWValidatorBase(const FWValidatorBase&) = delete; // stop default

   const FWValidatorBase& operator=(const FWValidatorBase&) = delete; // stop default

   // ---------- member data --------------------------------

};


#endif
