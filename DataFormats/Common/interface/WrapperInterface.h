#ifndef DataFormats_Common_WrapperInterface_h
#define DataFormats_Common_WrapperInterface_h

/*----------------------------------------------------------------------

WrapperInterface: 

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/WrapperInterfaceBase.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "boost/utility.hpp"
#include <vector>

namespace edm {

  template <typename T>
  class WrapperInterface : public WrapperInterfaceBase, private boost::noncopyable {
  public:
    typedef T value_type;
    typedef T wrapped_type;

    WrapperInterface() : WrapperInterfaceBase() {}
    virtual ~WrapperInterface() {}

    //these are used by FWLite
    static std::type_info const& productTypeInfo() {return typeid(T);}
    static std::type_info const& typeInfo() {return typeid(Wrapper<T>);}

  private:
    static Wrapper<T> const* typeCast(void const* me) {
      return static_cast<Wrapper<T> const*>(me);
    }

    static Wrapper<T>* typeCast(void* me) {
      return static_cast<Wrapper<T>*>(me);
    }

    virtual void deleteProduct_(void const* me) const {
      delete typeCast(me);
    }

    virtual void do_fillView(void const* me,
                             ProductID const& id,
                             std::vector<void const*>& pointers,
                             helper_vector_ptr& helpers) const {
      
      typeCast(me)->fillView(id, pointers, helpers);
    }
    virtual void do_setPtr(void const* me,
                           std::type_info const& iToType,
                           unsigned long iIndex,
                           void const*& oPtr) const {
      typeCast(me)->setPtr(iToType, iIndex, oPtr);
    }

    virtual void do_fillPtrVector(void const* me,
                                  std::type_info const& iToType,
                                  std::vector<unsigned long> const& iIndicies,
                                  std::vector<void const*>& oPtr) const {
      typeCast(me)->fillPtrVector(iToType, iIndicies, oPtr);
    }

    virtual bool isMergeable_(void const* me) const {
      return typeCast(me)->isMergeable();
    }

    virtual bool hasIsProductEqual_(void const* me) const {
      return typeCast(me)->hasIsProductEqual();
    }

    virtual bool mergeProduct_(void* me, void const* newProduct) const {
      return typeCast(me)->mergeProduct(typeCast(newProduct));
    }

    virtual bool isProductEqual_(void const* me, void const* newProduct) const {
      return typeCast(me)->isProductEqual(typeCast(newProduct));
    }

    virtual std::type_info const& dynamicTypeInfo_() const {
      return typeid(T);
    }

    virtual std::type_info const& wrappedTypeInfo_() const {
      return typeid(Wrapper<T>);
    }

    virtual bool isPresent_(void const* me) const {
      return typeCast(me)->isPresent();
    }
  };
} //namespace edm

#endif
