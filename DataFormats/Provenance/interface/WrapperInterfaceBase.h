#ifndef DataFormats_Provenance_WrapperInterfaceBase_h
#define DataFormats_Provenance_WrapperInterfaceBase_h

/*----------------------------------------------------------------------

WrapperInterfaceBase: The base class of all things that will be inserted into the Event.
/
----------------------------------------------------------------------*/
#include "boost/shared_ptr.hpp"

#include <typeinfo>
#include <vector>

namespace edm {
  class ProductID;
  namespace reftobase {
    class RefVectorHolderBase;
  }
  typedef boost::shared_ptr<reftobase::RefVectorHolderBase> helper_vector_ptr;
  class WrapperInterfaceBase {
  public:
    WrapperInterfaceBase();
    virtual ~WrapperInterfaceBase();

    void deleteProduct(void const* me) const {
      deleteProduct_(me);
    }

    void fillView(void const* me,
                  ProductID const& id,
                  std::vector<void const*>& view,
                  helper_vector_ptr& helpers) const;

    void setPtr(void const* me,
                std::type_info const& iToType,
                unsigned long iIndex,
                void const*& oPtr) const;

    void fillPtrVector(void const* me,
                       std::type_info const& iToType,
                       std::vector<unsigned long> const& iIndicies,
                       std::vector<void const*>& oPtr) const;

    bool isMergeable(void const* me) const {
      return isMergeable_(me);
    }

    bool hasIsProductEqual(void const* me) const {
      return hasIsProductEqual_(me);
    }

    bool mergeProduct(void* me, void const* newProduct) const {
      return mergeProduct_(me, newProduct);
    }

    bool isProductEqual(void const* me, void const* newProduct) const {
      return isProductEqual_(me, newProduct);
    }

    bool isPresent(void const* me) const {
      return isPresent_(me);
    }

    std::type_info const& dynamicTypeInfo() const {
      return dynamicTypeInfo_();
    }

    std::type_info const& wrappedTypeInfo() const {
      return wrappedTypeInfo_();
    }

  private:
    virtual void deleteProduct_(void const* me) const = 0;

    virtual void do_fillView(void const* me,
                             ProductID const& id,
                             std::vector<void const*>& pointers,
                             helper_vector_ptr& helpers) const = 0;

    virtual void do_setPtr(void const* me,
                           std::type_info const& iToType,
                           unsigned long iIndex,
                           void const*& oPtr) const = 0;

    virtual void do_fillPtrVector(void const* me,
                                  std::type_info const& iToType,
                                  std::vector<unsigned long> const& iIndicies,
                                  std::vector<void const*>& oPtr) const = 0;

    virtual bool isMergeable_(void const* me) const = 0;

    virtual bool hasIsProductEqual_(void const* me) const = 0;

    virtual bool mergeProduct_(void* me, void const* newProduct) const = 0;

    virtual bool isProductEqual_(void const* me, void const* newProduct) const = 0;

    virtual std::type_info const& dynamicTypeInfo_() const = 0;

    virtual std::type_info const& wrappedTypeInfo_() const = 0;

    virtual bool isPresent_(void const* me) const = 0;
  };
}
#endif
