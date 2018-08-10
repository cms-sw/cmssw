#ifndef DataFormats_Common_WrapperBase_h
#define DataFormats_Common_WrapperBase_h

/*----------------------------------------------------------------------

WrapperBase: The base class of all things that will be inserted into the Event.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Common/interface/FillViewHelperVector.h"
#include "DataFormats/Provenance/interface/ViewTypeChecker.h"

#include <typeinfo>
#include <vector>
#include <memory>

namespace edm {
  namespace soa {
    class TableExaminerBase;
  }
  
  class WrapperBase : public ViewTypeChecker {
  public:
    //used by inheriting classes to force construction via emplace
    struct Emplace {};

    WrapperBase();
    ~WrapperBase() override;
    bool isPresent() const {return isPresent_();}

    // We have to use vector<void*> to keep the type information out
    // of the WrapperBase class.
    void fillView(ProductID const& id,
                  std::vector<void const*>& view,
                  FillViewHelperVector& helpers) const;

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

    bool isMergeable() const {return isMergeable_();}
    bool mergeProduct(WrapperBase const* newProduct) {return mergeProduct_(newProduct);}
    bool hasIsProductEqual() const {return hasIsProductEqual_();}
    bool isProductEqual(WrapperBase const* newProduct) const {return isProductEqual_(newProduct);}
    bool hasSwap() const {return hasSwap_();}
    void swapProduct(WrapperBase* newProduct) {swapProduct_(newProduct);}

    std::shared_ptr<soa::TableExaminerBase> tableExaminer() const {
      return tableExaminer_();
    }
  private:
    virtual std::type_info const& dynamicTypeInfo_() const = 0;

    virtual std::type_info const& wrappedTypeInfo_() const = 0;

    // This will never be called.
    // For technical ROOT related reasons, we cannot
    // declare it = 0.
    virtual bool isPresent_() const {return true;}

    virtual bool isMergeable_() const = 0;
    virtual bool mergeProduct_(WrapperBase const* newProduct ) = 0;
    virtual bool hasIsProductEqual_() const = 0;
    virtual bool isProductEqual_(WrapperBase const* newProduct) const = 0;
    virtual bool hasSwap_() const = 0;
    virtual void swapProduct_(WrapperBase* newProduct) = 0;

    virtual void do_fillView(ProductID const& id,
                             std::vector<void const*>& pointers,
                             FillViewHelperVector & helpers) const = 0;
    virtual void do_setPtr(std::type_info const& iToType,
                           unsigned long iIndex,
                           void const*& oPtr) const = 0;

    virtual void do_fillPtrVector(std::type_info const& iToType,
                                  std::vector<unsigned long> const& iIndicies,
                                  std::vector<void const*>& oPtr) const = 0;
    
    virtual std::shared_ptr<soa::TableExaminerBase> tableExaminer_() const = 0;

  };
}
#endif
