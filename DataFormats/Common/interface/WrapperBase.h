#ifndef DataFormats_Common_WrapperBase_h
#define DataFormats_Common_WrapperBase_h

/*----------------------------------------------------------------------

WrapperBase: The base class of all things that will be inserted into the Event.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Provenance/interface/ViewTypeChecker.h"

#include <typeinfo>
#include <vector>

namespace edm {
  class WrapperBase : public ViewTypeChecker {
  public:
    WrapperBase();
    virtual ~WrapperBase();
    bool isPresent() const {return isPresent_();}

    // We have to use vector<void*> to keep the type information out
    // of the WrapperBase class.
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

    bool sameType(WrapperBase const& other) const {
      return other.dynamicTypeInfo() == dynamicTypeInfo();
    }

#ifndef __GCCXML__
    bool isMergeable() const {return isMergeable_();}
    bool mergeProduct(WrapperBase const* newProduct) {return mergeProduct_(newProduct);}
    bool hasIsProductEqual() const {return hasIsProductEqual_();}
    bool isProductEqual(WrapperBase const* newProduct) const {return isProductEqual_(newProduct);}
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
    virtual bool mergeProduct_(WrapperBase const* newProduct) { return true; }
    virtual bool hasIsProductEqual_() const { return true; }
    virtual bool isProductEqual_(WrapperBase const* newProduct) const { return true; }
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
