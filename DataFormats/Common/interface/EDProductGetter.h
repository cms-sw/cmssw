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
  namespace detail {
    using GetThinnedKeyFromExceptionFactory = std::function<edm::Exception()>;
  }
  using OptionalThinnedKey = std::variant<unsigned int, detail::GetThinnedKeyFromExceptionFactory, std::monostate>;

  class EDProductGetter {
  public:
    EDProductGetter();
    virtual ~EDProductGetter();

    EDProductGetter(EDProductGetter const&) = delete;  // stop default

    EDProductGetter const& operator=(EDProductGetter const&) = delete;  // stop default

    // ---------- const member functions ---------------------
    virtual WrapperBase const* getIt(ProductID const&) const = 0;

    // getThinnedProduct assumes getIt was already called and failed to find
    // the product. The input key is the index of the desired element in the
    // container identified by ProductID (which cannot be found).
    // If the return value is not null, then the desired element was
    // found in a thinned container. If the desired element is not
    // found, then an optional without a value is returned.
    virtual std::optional<std::tuple<WrapperBase const*, unsigned int>> getThinnedProduct(ProductID const&,
                                                                                          unsigned int key) const = 0;

    // getThinnedProducts assumes getIt was already called and failed to find
    // the product. The input keys are the indexes into the container identified
    // by ProductID (which cannot be found). On input the WrapperBase pointers
    // must all be set to nullptr (except when the function calls itself
    // recursively where non-null pointers mark already found elements).
    // Thinned containers derived from the product are searched to see
    // if they contain the desired elements. For each that is
    // found, the corresponding WrapperBase pointer is set and the key
    // is modified to be the key into the container where the element
    // was found. The WrapperBase pointers might or might not all point
    // to the same thinned container.
    virtual void getThinnedProducts(ProductID const& pid,
                                    std::vector<WrapperBase const*>& foundContainers,
                                    std::vector<unsigned int>& keys) const = 0;

    // This overload is allowed to be called also without getIt()
    // being called first, but the thinned ProductID must come from an
    // existing RefCore. The input key is the index of the desired
    // element in the container identified by the parent ProductID.
    // Returns an std::variant whose contents can be
    // - unsigned int for the index in the thinned collection if the
    //   desired element was found in the thinned collection
    // - function creating an edm::Exception if parent is not a parent
    //   of any thinned collection, thinned is not really a thinned
    //   collection, or parent and thinned have no thinning
    //   relationship
    // - std::monostate if thinned is thinned from parent, but the key
    //   is not found in the thinned collection
    virtual OptionalThinnedKey getThinnedKeyFrom(ProductID const& parent,
                                                 unsigned int key,
                                                 ProductID const& thinned) const = 0;

    unsigned int transitionIndex() const { return transitionIndex_(); }

    // ---------- member functions ---------------------------

    ///These can only be used internally by the framework
    static EDProductGetter const* switchProductGetter(EDProductGetter const*);
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
