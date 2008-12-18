#ifndef DataFormats_Common_EDProduct_h
#define DataFormats_Common_EDProduct_h

/*----------------------------------------------------------------------
  
EDProduct: The base class of all things that will be inserted into the
Event.

----------------------------------------------------------------------*/

#include <vector>
#include "DataFormats/Common/interface/EDProductfwd.h"

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
		  helper_vector_ptr & helpers) const;
    
    void setPtr(const std::type_info& iToType,
                unsigned long iIndex,
                void const*& oPtr) const;

    void fillPtrVector(const std::type_info& iToType,
                          const std::vector<unsigned long>& iIndicies,
                          std::vector<void const*>& oPtr) const;

#ifndef __REFLEX__
    bool isMergeable() const {return isMergeable_();}
    bool mergeProduct(EDProduct const* newProduct) {return mergeProduct_(newProduct);}
    bool hasIsProductEqual() const {return hasIsProductEqual_();}
    bool isProductEqual(EDProduct const* newProduct) const {return isProductEqual_(newProduct);}
#endif

  private:
    // These will never be called.
    // For technical ROOT related reasons, we cannot
    // declare it = 0.
    virtual bool isPresent_() const {return true;}

#ifndef __REFLEX__
    virtual bool isMergeable_() const { return true; }
    virtual bool mergeProduct_(EDProduct const* newProduct) { return true; }
    virtual bool hasIsProductEqual_() const { return true; }
    virtual bool isProductEqual_(EDProduct const* newProduct) const { return true; }
#endif

    virtual void do_fillView(ProductID const& id,
			     std::vector<void const*>& pointers,
			     helper_vector_ptr & helpers) const = 0;
    virtual void do_setPtr(const std::type_info& iToType,
                           unsigned long iIndex,
                           void const*& oPtr) const = 0;

    virtual void do_fillPtrVector(const std::type_info& iToType,
                                     const std::vector<unsigned long>& iIndicies,
                                     std::vector<void const*>& oPtr) const = 0;
  };
}
#endif
