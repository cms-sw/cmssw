#ifndef DataFormats_Common_EDProduct_h
#define DataFormats_Common_EDProduct_h

/*----------------------------------------------------------------------

EDProduct: The base class of all things that will be inserted into the Event.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProductfwd.h"

#include <typeinfo>
#include <vector>

namespace edm {
  class EDProduct {
  public:
    EDProduct();
    virtual ~EDProduct();
    bool isPresent() const {return isPresent_();}

    // We have to use vector<void*> to keep the type information out
    // of the EDProduct class.
    void fillView(ProductID const& id,
                  std::vector<void const*>& view,
                  helper_vector_ptr& helpers) const;

    void setPtr(std::type_info const& iToType,
                unsigned long iIndex,
                void const*& oPtr) const;

    void fillPtrVector(std::type_info const& iToType,
                          std::vector<unsigned long> const& iIndicies,
                          std::vector<void const*>& oPtr) const;

    std::type_info const& dynamicTypeInfo() const {return dynamicTypeInfo_();}

    std::type_info const& wrappedTypeInfo() const {return wrappedTypeInfo_();}

    bool sameType(EDProduct const& other) const {
      return other.dynamicTypeInfo() == dynamicTypeInfo();
    }

#ifndef __GCCXML__
    bool isMergeable() const {return isMergeable_();}
    bool mergeProduct(EDProduct const* newProduct) {return mergeProduct_(newProduct);}
    bool hasIsProductEqual() const {return hasIsProductEqual_();}
    bool isProductEqual(EDProduct const* newProduct) const {return isProductEqual_(newProduct);}
#endif

  private:
    virtual std::type_info const& dynamicTypeInfo_() const = 0;

    virtual std::type_info const& wrappedTypeInfo_() const = 0;

    // This will never be called.
    // For technical ROOT related reasons, we cannot
    // declare it = 0.
    virtual bool isPresent_() const {return true;}

#ifndef __GCCXML__
    virtual bool isMergeable_() const { return true; }
    virtual bool mergeProduct_(EDProduct const* newProduct) { return true; }
    virtual bool hasIsProductEqual_() const { return true; }
    virtual bool isProductEqual_(EDProduct const* newProduct) const { return true; }
#endif

    virtual void do_fillView(ProductID const& id,
                             std::vector<void const*>& pointers,
                             helper_vector_ptr & helpers) const = 0;
    virtual void do_setPtr(std::type_info const& iToType,
                           unsigned long iIndex,
                           void const*& oPtr) const = 0;

    virtual void do_fillPtrVector(std::type_info const& iToType,
                                  std::vector<unsigned long> const& iIndicies,
                                  std::vector<void const*>& oPtr) const = 0;
  };
}
#endif
