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

// system include files
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

// forward declarations

namespace edm {

  class Exception;
  class ProductID;
  class WrapperBase;
  class EDProductGetter {
  public:
    EDProductGetter();
    virtual ~EDProductGetter();

    EDProductGetter(EDProductGetter const&) = delete;  // stop default

    EDProductGetter const& operator=(EDProductGetter const&) = delete;  // stop default

    // ---------- const member functions ---------------------
    virtual WrapperBase const* getIt(ProductID const&) const = 0;

    unsigned int transitionIndex() const { return transitionIndex_(); }

    // ---------- member functions ---------------------------

    ///These can only be used internally by the framework
    static EDProductGetter const* switchProductGetter(EDProductGetter const*);
    static void setMultiThreadProductGetter(EDProductGetter const*);
    static void unsetMultiThreadProductGetter();
    static void assignEDProductGetter(EDProductGetter const*&);

  private:
    virtual unsigned int transitionIndex_() const = 0;

    // ---------- member data --------------------------------
  };

  EDProductGetter const* mustBeNonZero(EDProductGetter const* prodGetter,
                                       std::string refType,
                                       ProductID const& productID);
}  // namespace edm
#endif
