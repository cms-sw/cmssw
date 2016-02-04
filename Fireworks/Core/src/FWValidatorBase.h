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
// $Id: FWValidatorBase.h,v 1.3 2009/01/23 21:35:44 amraktad Exp $
//

// system include files
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>

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
                            std::vector<std::pair<boost::shared_ptr<std::string>, std::string> >& oOptions) const = 0;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWValidatorBase(const FWValidatorBase&); // stop default

   const FWValidatorBase& operator=(const FWValidatorBase&); // stop default

   // ---------- member data --------------------------------

};


#endif
