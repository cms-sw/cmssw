#ifndef DataFormats_Common_EDProductGetter_h
#define DataFormats_Common_EDProductGetter_h
// -*- C++ -*-
//
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
//

// user include files
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Provenance/interface/ProductID.h"

// system include files
#include "boost/utility.hpp"

// forward declarations

namespace edm {
   class EDProductGetter : private boost::noncopyable {

   public:

     EDProductGetter();
     virtual ~EDProductGetter();

     // ---------- const member functions ---------------------
     virtual WrapperBase const* getIt(ProductID const&) const = 0;

     unsigned int transitionIndex() const {
       return transitionIndex_();
     }

     // ---------- member functions ---------------------------

     ///These can only be used internally by the framework
     static EDProductGetter const* switchProductGetter(EDProductGetter const*);
     static void assignEDProductGetter(EDProductGetter const*&);

private:
    virtual unsigned int transitionIndex_() const = 0;

     // ---------- member data --------------------------------

   };

   EDProductGetter const*
   mustBeNonZero(EDProductGetter const* prodGetter, std::string refType, ProductID const& productID);
}
#endif
