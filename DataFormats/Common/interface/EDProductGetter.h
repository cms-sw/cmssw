#ifndef Common_EDProductGetter_h
#define Common_EDProductGetter_h
// -*- C++ -*-
//
// Package:     EDProduct
// Class  :     EDProductGetter
// 
/**\class EDProductGetter EDProductGetter.h DataFormats/Common/interface/EDProductGetter.h

 Description: Abstract base class used internally by the RefBase to obtain the EDProduct from the Event

 Usage:
    This is used internally by the edm::Ref classes.

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Nov  1 15:06:31 EST 2005
// $Id: EDProductGetter.h,v 1.2 2006/02/08 00:44:23 wmtan Exp $
//

// system include files

// user include files
#include "DataFormats/Common/interface/ProductID.h"

// forward declarations

namespace edm {
   class EDProduct;
   class EDProductGetter {

   public:
   
      EDProductGetter();
      virtual ~EDProductGetter();

      // ---------- const member functions ---------------------
      virtual EDProduct const* getIt(ProductID const&) const = 0;
      
      // ---------- static member functions --------------------
      static EDProductGetter const* instance();
      
      // ---------- member functions ---------------------------

      ///Helper class to make the EDProductGetter accessible on at the proper times
      class Operate {
       public:
         Operate(EDProductGetter const* iGet) : previous_(EDProductGetter::set(iGet)) {
         }
         ~Operate() {
            EDProductGetter::set(previous_);
         }
       private:
         EDProductGetter const* previous_;
      };
      
      friend class Operate;
private:
      EDProductGetter(EDProductGetter const&); // stop default

      const EDProductGetter& operator=(EDProductGetter const&); // stop default

      /**This does not take ownership of the argument, so it is up to the caller to be
         sure that the object lifetime is greater than the time for which it is set*/
      static EDProductGetter const* set(EDProductGetter const*);
      // ---------- member data --------------------------------
      
   };
}

#endif
