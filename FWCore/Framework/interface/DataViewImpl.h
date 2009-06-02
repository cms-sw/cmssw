#ifndef Framework_DataViewImpl_h
#define Framework_DataViewImpl_h

// -*- C++ -*-
//

// Class  :     DataViewImpl
// 
/**\class DataViewImpl DataViewImpl.h FWCore/Framework/interface/DataViewImpl.h

Description: This is the implementation for accessing EDProducts and 
inserting new EDproducts.

Usage:

Getting Data

The edm::DataViewImpl class provides many 'get*" methods for getting data
it contains.  

The primary method for getting data is to use getByLabel(). The labels are
the label of the module assigned in the configuration file and the 'product
instance label' (which can be omitted in the case the 'product instance label'
is the default value).  The C++ type of the product plus the two labels
uniquely identify a product in the DataViewImpl.

We use an event in the examples, but a run or a luminosity block can also
hold products.

\code
edm::Handle<AppleCollection> apples;
event.getByLabel("tree",apples);
\endcode

\code
edm::Handle<FruitCollection> fruits;
event.getByLabel("market", "apple", fruits);
\endcode


Putting Data

\code
std::auto_ptr<AppleCollection> pApples( new AppleCollection );
  
//fill the collection
...
event.put(pApples);
\endcode

\code
std::auto_ptr<FruitCollection> pFruits( new FruitCollection );

//fill the collection
...
event.put("apple", pFruits);
\endcode


Getting a reference to a product before that product is put into the
event/lumiBlock/run.
NOTE: The edm::RefProd returned will not work until after the
edm::DataViewImpl has been committed (which happens after the
EDProducer::produce method has ended)
\code
std::auto_ptr<AppleCollection> pApples( new AppleCollection);

edm::RefProd<AppleCollection> refApples = event.getRefBeforePut<AppleCollection>();

//do loop and fill collection
for(unsigned int index = 0; ..... ) {
....
apples->push_back( Apple(...) );
  
//create an edm::Ref to the new object
edm::Ref<AppleCollection> ref(refApples, index);
....
}
\endcode

*/
/*----------------------------------------------------------------------

----------------------------------------------------------------------*/
#include <cassert>
#include <memory>
#include <typeinfo>
#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "boost/type_traits.hpp"

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Common/interface/traits.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/ConvertHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {

  class DataViewImpl {
  public:
    DataViewImpl(Principal & pcpl,
		 ModuleDescription const& md,
		 BranchType const& branchType);

    ~DataViewImpl();

    size_t size() const;

    template <typename PROD>
    bool 
    get(SelectorBase const&, Handle<PROD>& result) const;
  
    template <typename PROD>
    bool 
    getByLabel(std::string const& label, Handle<PROD>& result) const;

    template <typename PROD>
    bool 
    getByLabel(std::string const& label,
	       std::string const& productInstanceName, 
	       Handle<PROD>& result) const;

    /// same as above, but using the InputTag class 	 
    template <typename PROD> 	 
    bool 	 
    getByLabel(InputTag const& tag, Handle<PROD>& result) const; 	 

    template <typename PROD>
    void 
    getMany(SelectorBase const&, std::vector<Handle<PROD> >& results) const;

    template <typename PROD>
    bool
    getByType(Handle<PROD>& result) const;

    template <typename PROD>
    void 
    getManyByType(std::vector<Handle<PROD> >& results) const;

    ProcessHistory const&
    processHistory() const;

    DataViewImpl const&
    me() const {return *this;}

    typedef std::vector<std::pair<boost::shared_ptr<EDProduct>, ConstBranchDescription const*> >  ProductPtrVec;
  protected:

    Principal& principal() {return principal_;}
    Principal const& principal() const {return principal_;}

    ProductPtrVec & putProducts() {return putProducts_;}
    ProductPtrVec const& putProducts() const {return putProducts_;}

    ProductPtrVec & putProductsWithoutParents() {return putProductsWithoutParents_;}
    ProductPtrVec const& putProductsWithoutParents() const {return putProductsWithoutParents_;}
    
    ConstBranchDescription const&
    getBranchDescription(TypeID const& type, std::string const& productInstanceName) const;

    typedef std::vector<BasicHandle>  BasicHandleVec;

    //------------------------------------------------------------
    // Protected functions.
    //

    // The following 'get' functions serve to isolate the DataViewImpl class
    // from the Principal class.

    BasicHandle 
    get_(TypeID const& tid, SelectorBase const&) const;
    
    BasicHandle 
    getByLabel_(TypeID const& tid,
		std::string const& label,
		std::string const& productInstanceName,
		std::string const& processName) const;

    BasicHandle 
    getByLabel_(TypeID const& tid,
		std::string const& label,
		std::string const& productInstanceName,
		std::string const& processName,
		TypeID& typeID,
		size_t& cachedOffset,
		int& fillCount) const;

    void 
    getMany_(TypeID const& tid, 
	     SelectorBase const& sel, 
	     BasicHandleVec& results) const;

    BasicHandle 
    getByType_(TypeID const& tid) const;

    void 
    getManyByType_(TypeID const& tid, 
		   BasicHandleVec& results) const;

    int 
    getMatchingSequence_(TypeID const& typeID,
                         SelectorBase const& selector,
                         BasicHandle& result) const;

    int 
    getMatchingSequenceByLabel_(TypeID const& typeID,
                                std::string const& label,
                                std::string const& productInstanceName,
                                BasicHandle& result) const;

    int 
    getMatchingSequenceByLabel_(TypeID const& typeID,
                                std::string const& label,
                                std::string const& productInstanceName,
                                std::string const& processName,
                                BasicHandle& result) const;
    
  protected:
    // Also isolates the DataViewImpl class
    // from the Principal class.
    EDProductGetter const* prodGetter() const;
  private:
    //------------------------------------------------------------
    // Copying and assignment of DataViewImpls is disallowed
    //
    DataViewImpl(DataViewImpl const&);                  // not implemented
    DataViewImpl const& operator=(DataViewImpl const&);   // not implemented

  private:
    //------------------------------------------------------------
    // Data members
    //

    // putProducts_ and putProductsWithoutParents_ are the holding
    // pens for EDProducts inserted into this DataViewImpl. Pointers
    // in these collections own the products to which they point.
    // 
    ProductPtrVec putProducts_;               // keep parentage info for these
    ProductPtrVec putProductsWithoutParents_; // ... but not for these

    // Each DataViewImpl must have an associated Principal, used as the
    // source of all 'gets' and the target of 'puts'.
    Principal & principal_;

    // Each DataViewImpl must have a description of the module executing the
    // "transaction" which the DataViewImpl represents.
    ModuleDescription const& md_;

    // Is this an Event, a LuminosityBlock, or a Run.
    BranchType const branchType_;
  };

  template <typename PROD>
  inline
  std::ostream& 
  operator<<(std::ostream& os, Handle<PROD> const& h) {
    os << h.product() << " " << h.provenance() << " " << h.id();
    return os;
  }


  //------------------------------------------------------------
  // Metafunction support for compile-time selection of code used in
  // DataViewImpl::put member template.
  //

  // has_postinsert is a metafunction of one argument, the type T.  As
  // with many metafunctions, it is implemented as a class with a data
  // member 'value', which contains the value 'returned' by the
  // metafunction.
  //
  // has_postinsert<T>::value is 'true' if T has the post_insert
  // member function (with the right signature), and 'false' if T has
  // no such member function.

  namespace detail {
    typedef char (& no_tag )[1]; // type indicating FALSE
    typedef char (& yes_tag)[2]; // type indicating TRUE

    // Definitions forthe following struct and function templates are
    // not needed; we only require the declarations.
    template <typename T, void (T::*)()>  struct postinsert_function;
    template <typename T> no_tag  has_postinsert_helper(...);
    template <typename T> yes_tag has_postinsert_helper(postinsert_function<T, &T::post_insert> * p);


    template<typename T>
    struct has_postinsert {
      static bool const value = 
	sizeof(has_postinsert_helper<T>(0)) == sizeof(yes_tag) &&
	!boost::is_base_of<DoNotSortUponInsertion, T>::value;
    };


    // has_donotrecordparents<T>::value is true if we should not
    // record parentage for type T, and false otherwise.

    template <typename T>
    struct has_donotrecordparents {
      static bool const value = 
	boost::is_base_of<DoNotRecordParents,T>::value;
    };

  }

  //------------------------------------------------------------

  // The following function objects are used by Event::put, under the
  // control of a metafunction if, to either call the given object's
  // post_insert function (if it has one), or to do nothing (if it
  // does not have a post_insert function).
  template <typename T>
  struct DoPostInsert {
    void operator()(T* p) const { p->post_insert(); }
  };

  template <typename T>
  struct DoNotPostInsert {
    void operator()(T*) const { }
  };

  // Implementation of  DataViewImpl  member templates. See  DataViewImpl.cc for the
  // implementation of non-template members.
  //

  template <typename PROD>
  inline
  bool 
  DataViewImpl::get(SelectorBase const& sel,
		    Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = this->get_(TypeID(typeid(PROD)),sel);
    convert_handle(bh, result);  // throws on conversion error
    if (bh.failedToGet()) {
      return false;
    }
    return true;
  }
  
  template <typename PROD>
  inline
  bool
  DataViewImpl::getByLabel(std::string const& label,
			   Handle<PROD>& result) const {
    result.clear();
    return getByLabel(label, std::string(), result);
  }

  template <typename PROD>
  inline
  bool
  DataViewImpl::getByLabel(InputTag const& tag, Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = this->getByLabel_(TypeID(typeid(PROD)), tag.label(), tag.instance(), tag.process(), tag.typeID(), tag.cachedOffset(), tag.fillCount());
    convert_handle(bh, result);  // throws on conversion error
    if (bh.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename PROD>
  inline
  bool
  DataViewImpl::getByLabel(std::string const& label,
			   std::string const& productInstanceName,
			   Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = this->getByLabel_(TypeID(typeid(PROD)), label, productInstanceName, std::string());
    convert_handle(bh, result);  // throws on conversion error
    if (bh.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename PROD>
  inline
  void 
  DataViewImpl::getMany(SelectorBase const& sel,
			std::vector<Handle<PROD> >& results) const { 
    BasicHandleVec bhv;
    this->getMany_(TypeID(typeid(PROD)), sel, bhv);
    
    // Go through the returned handles; for each element,
    //   1. create a Handle<PROD> and
    //
    // This function presents an exception safety difficulty. If an
    // exception is thrown when converting a handle, the "got
    // products" record will be wrong.
    //
    // Since EDProducers are not allowed to use this function,
    // the problem does not seem too severe.
    //
    // Question: do we even need to keep track of the "got products"
    // for this function, since it is *not* to be used by EDProducers?
    std::vector<Handle<PROD> > products;

    typename BasicHandleVec::const_iterator it = bhv.begin();
    typename BasicHandleVec::const_iterator end = bhv.end();

    while (it != end) {
      Handle<PROD> result;
      convert_handle(*it, result);  // throws on conversion error
      products.push_back(result);
      ++it;
    }
    results.swap(products);
  }

  template <typename PROD>
  inline
  bool
  DataViewImpl::getByType(Handle<PROD>& result) const {
    result.clear();
    BasicHandle bh = this->getByType_(TypeID(typeid(PROD)));
    convert_handle(bh, result);  // throws on conversion error
    if (bh.failedToGet()) {
      return false;
    }
    return true;
  }

  template <typename PROD>
  inline
  void 
  DataViewImpl::getManyByType(std::vector<Handle<PROD> >& results) const { 
    BasicHandleVec bhv;
    this->getManyByType_(TypeID(typeid(PROD)), bhv);
    
    // Go through the returned handles; for each element,
    //   1. create a Handle<PROD> and
    //
    // This function presents an exception safety difficulty. If an
    // exception is thrown when converting a handle, the "got
    // products" record will be wrong.
    //
    // Since EDProducers are not allowed to use this function,
    // the problem does not seem too severe.
    //
    // Question: do we even need to keep track of the "got products"
    // for this function, since it is *not* to be used by EDProducers?
    std::vector<Handle<PROD> > products;

    typename BasicHandleVec::const_iterator it = bhv.begin();
    typename BasicHandleVec::const_iterator end = bhv.end();

    while (it != end) {
      Handle<PROD> result;
      convert_handle(*it, result);  // throws on conversion error
      products.push_back(result);
      ++it;
    }
    results.swap(products);
  }
}
#endif
