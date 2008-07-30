#ifndef Framework_DataViewImpl_h
#define Framework_DataViewImpl_h

// -*- C++ -*-
//
// Package:     Framework
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

$Id: DataViewImpl.h,v 1.37 2008/03/31 21:13:27 wmtan Exp $

----------------------------------------------------------------------*/
#include <cassert>
#include <memory>
#include <typeinfo>
#include <set>
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
#include "DataFormats/Common/interface/traits.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace edm {

  class DataViewImpl {
  public:
    DataViewImpl(Principal & dbk,
		 ModuleDescription const& md,
		 BranchType const& branchType);

    ~DataViewImpl();

    size_t size() const;

    template <typename PROD>
    bool 
    get(ProductID const& oid, Handle<PROD>& result) const;

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

    // Template member overload to deal with Views. Perhaps only this
    // one needs to be overloaded, because the other getByLabel
    // implementations go through this one.
    template <typename ELEMENT>
    bool
    getByLabel(std::string const& label, 
	       std::string const& productInstanceName,
	       Handle<View<ELEMENT> >& result) const;

    // Template member overload to deal with Views.     
    template <typename ELEMENT>
    bool
    get(ProductID const& oid, Handle<View<ELEMENT> >& result) const ;

    /// same as above, but using the InputTag class 	 
    template <typename PROD> 	 
    bool 	 
    getByLabel(InputTag const& tag, Handle<PROD>& result) const; 	 

    template <typename ELEMENT> 	 
    bool 	 
    getByLabel(InputTag const& tag, Handle<View<ELEMENT> >& result) const; 	 
    
    template <typename PROD>
    void 
    getMany(SelectorBase const&, std::vector<Handle<PROD> >& results) const;

    template <typename PROD>
    bool
    getByType(Handle<PROD>& result) const;

    template <typename PROD>
    void 
    getManyByType(std::vector<Handle<PROD> >& results) const;

    Provenance const&
    getProvenance(ProductID const& theID) const;

    void
    getAllProvenance(std::vector<Provenance const*> &provenances) const;

    ///Put a new product.
    template <typename PROD>
    OrphanHandle<PROD>
    put(std::auto_ptr<PROD> product) {return put<PROD>(product, std::string());}

    ///Put a new product with a 'product instance name'
    template <typename PROD>
    OrphanHandle<PROD>
    put(std::auto_ptr<PROD> product, std::string const& productInstanceName);

    ///Returns a RefProd to a product before that product has been placed into the DataViewImpl
    /// The RefProd (and any Ref's made from it) will no work properly until after the
    /// DataViewImpl has been committed (which happens after leaving the EDProducer::produce method)
    template <typename PROD>
    RefProd<PROD>
    getRefBeforePut() {return getRefBeforePut<PROD>(std::string());}

    template <typename PROD>
    RefProd<PROD>
    getRefBeforePut(std::string const& productInstanceName);

    ProcessHistory const& processHistory() const;

    DataViewImpl const& me() const {return *this;}
    
  private:

    typedef std::set<ProductID>       ProductIDSet;
    typedef std::vector<std::pair<EDProduct*, ConstBranchDescription const *> >  ProductPtrVec;
    typedef std::vector<BasicHandle>  BasicHandleVec;

    //------------------------------------------------------------
    // Private functions.
    //

    ConstBranchDescription const&
    getBranchDescription(TypeID const& type, std::string const& productInstanceName) const;

    // commit_ is called to complete the transaction represented by
    // this DataViewImpl. The friendships required seems gross, but any
    // alternative is not great either.  Putting it into the
    // public interface is asking for trouble
    friend class ConfigurableInputSource;
    friend class DaqSource;
    friend class RawInputSource;
    friend class InputSource;
    friend class EDFilter;
    friend class EDProducer;

    void commit_();

    // The following 'get' functions serve to isolate the DataViewImpl class
    // from the Principal class.

    BasicHandle 
    get_(ProductID const& oid) const;

    BasicHandle 
    get_(TypeID const& tid, SelectorBase const&) const;
    
    BasicHandle 
    getByLabel_(TypeID const& tid,
		std::string const& label,
		std::string const& productInstanceName) const;

    BasicHandle 
    getByLabel_(TypeID const& tid,
		std::string const& label,
		std::string const& productInstanceName,
		std::string const& processName) const;

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
                         BasicHandleVec& results,
                         bool stopIfProcessHasMatch) const;

    int 
    getMatchingSequenceByLabel_(TypeID const& typeID,
                                std::string const& label,
                                std::string const& productInstanceName,
                                BasicHandleVec& results,
                                bool stopIfProcessHasMatch) const;

    int 
    getMatchingSequenceByLabel_(TypeID const& typeID,
                                std::string const& label,
                                std::string const& productInstanceName,
                                std::string const& processName,
                                BasicHandleVec& results,
                                bool stopIfProcessHasMatch) const;
    
    template <typename ELEMENT>
    void
    fillView_(BasicHandle & bh,
	      Handle<View<ELEMENT> >& result) const;

    // Also isolates the DataViewImpl class
    // from the Principal class.
    EDProductGetter const* prodGetter() const;
    //------------------------------------------------------------
    // Copying and assignment of DataViewImpls is disallowed
    //
    DataViewImpl(DataViewImpl const&);                  // not implemented
    DataViewImpl const& operator=(DataViewImpl const&);   // not implemented

    //------------------------------------------------------------
    // Data members
    //

    // put_products_ is the holding pen for EDProducts inserted into
    // this DataViewImpl. Pointers in this collection own the products to
    // which they point.
    ProductPtrVec put_products_;

    // gotProductIDs_ must be mutable because it records all 'gets',
    // which do not logically modify the DataViewImpl. gotProductIDs_ is
    // merely a cache reflecting what has been retreived from the
    // Principal class.
    mutable ProductIDSet gotProductIDs_;

    // Each DataViewImpl must have an associated Principal, used as the
    // source of all 'gets' and the target of 'puts'.
    Principal & dbk_;

    // Each DataViewImpl must have a description of the module executing the
    // "transaction" which the DataViewImpl represents.
    ModuleDescription const& md_;

    // Is this an Event, a LuminosityBlock, or a Run.
    BranchType const branchType_;

    // We own the retrieved Views, and have to destroy them.
    mutable std::vector<boost::shared_ptr<ViewBase> > gotViews_;
  };

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

  namespace detail 
  {
    //------------------------------------------------------------
    // WHEN WE MOVE to a newer compiler version, the following code
    // should be activated. This code causes compilation failures under
    // GCC 3.2.3, because of a compiler error in dealing with our
    // application of SFINAE. GCC 3.4.2 is known to deal with this code
    // correctly.
    //------------------------------------------------------------
    typedef char (& no_tag )[1]; // type indicating FALSE
    typedef char (& yes_tag)[2]; // type indicating TRUE

    // Definitions forthe following struct and function templates are
    // not needed; we only require the declarations.
    template <typename T, void (T::*)()>  struct postinsert_function;
    template <typename T> no_tag  has_postinsert_helper(...);
    template <typename T> yes_tag has_postinsert_helper(postinsert_function<T, &T::post_insert> * p);


    template<typename T>
    struct has_postinsert
    {
      static bool const value = 
	sizeof(has_postinsert_helper<T>(0)) == sizeof(yes_tag) &&
	!boost::is_base_of<edm::DoNotSortUponInsertion, T>::value;
    };
  }



  //------------------------------------------------------------

  // The following function objects are used by DataViewImpl::put, under the
  // control of a metafunction if, to either call the given object's
  // post_insert function (if it has one), or to do nothing (if it
  // does not have a post_insert function).
  template <typename T>
  struct DoPostInsert
  {
    void operator()(T* p) const { p->post_insert(); }
  };

  template <typename T>
  struct DoNotPostInsert
  {
    void operator()(T*) const { }
  };


  //------------------------------------------------------------
  //
  // Implementation of  DataViewImpl  member templates. See  DataViewImpl.cc for the
  // implementation of non-template members.
  //

  template <typename PROD>
  OrphanHandle<PROD> 
  DataViewImpl::put(std::auto_ptr<PROD> product, std::string const& productInstanceName)
  {
    if (product.get() == 0) {                // null pointer is illegal
      TypeID typeID(typeid(PROD));
      throw edm::Exception(edm::errors::NullPointerError)
        << "DataViewImpl::put: A null auto_ptr was passed to 'put'.\n"
	<< "The pointer is of type " << typeID << ".\n"
	<< "The specified productInstanceName was '" << productInstanceName << "'.\n";
    }

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    typename boost::mpl::if_c<detail::has_postinsert<PROD>::value, 
      DoPostInsert<PROD>, 
      DoNotPostInsert<PROD> >::type maybe_inserter;
    maybe_inserter(product.get());

    ConstBranchDescription const& desc =
      getBranchDescription(TypeID(*product), productInstanceName);

    Wrapper<PROD> *wp(new Wrapper<PROD>(product));

    put_products_.push_back(std::make_pair(wp, &desc));

    // product.release(); // The object has been copied into the Wrapper.
    // The old copy must be deleted, so we cannot release ownership.

    return(OrphanHandle<PROD>(wp->product(), desc.productID()));
  }

  template <typename PROD>
  RefProd<PROD>
  DataViewImpl::getRefBeforePut(std::string const& productInstanceName) {
    PROD* p = 0;
    ConstBranchDescription const& desc =
      getBranchDescription(TypeID(*p), productInstanceName);

    //should keep track of what Ref's have been requested and make sure they are 'put'
    return RefProd<PROD>(desc.productID(), prodGetter());
  }
  
  template <typename PROD>
  bool
  DataViewImpl::get(ProductID const& oid, Handle<PROD>& result) const
  {
    result.clear();
    BasicHandle bh = this->get_(oid);
    convert_handle(bh, result);  // throws on conversion error
    if(!bh.failedToGet()) {
      gotProductIDs_.insert(bh.id());
      return true;
    }
    return false;
  }

  template <typename PROD>
  bool 
  DataViewImpl::get(SelectorBase const& sel,
		    Handle<PROD>& result) const
  {
    result.clear();
    BasicHandle bh = this->get_(TypeID(typeid(PROD)),sel);
    convert_handle(bh, result);  // throws on conversion error
    if(!bh.failedToGet()) {
      gotProductIDs_.insert(bh.id());
      return true;
    }
    return false;
  }
  
  template <typename PROD>
  inline
  bool
  DataViewImpl::getByLabel(std::string const& label,
			   Handle<PROD>& result) const
  {
    result.clear();
    return getByLabel(label, std::string(), result);
  }

  template <typename PROD>
  bool
  DataViewImpl::getByLabel(InputTag const& tag, Handle<PROD>& result) const
  {
    result.clear();
    if (tag.process().empty()) {
      return getByLabel(tag.label(), tag.instance(), result);
    } else {
      BasicHandle bh = this->getByLabel_(TypeID(typeid(PROD)), tag.label(), tag.instance(),tag.process());
      convert_handle(bh, result);  // throws on conversion error
      if(!bh.failedToGet()) {
        gotProductIDs_.insert(bh.id());
        return true;
      }
    }
    return false;
  }

  template <typename PROD>
  bool
  DataViewImpl::getByLabel(std::string const& label,
			   std::string const& productInstanceName,
			   Handle<PROD>& result) const
  {
    result.clear();
    BasicHandle bh = this->getByLabel_(TypeID(typeid(PROD)), label, productInstanceName);
    convert_handle(bh, result);  // throws on conversion error
    if(!bh.failedToGet()) {
      gotProductIDs_.insert(bh.id());
      return true;
    }
    return false;
  }

  template <class T>
  std::ostream& 
  operator<<(std::ostream& os, Handle<T> const& h)
  {
    os << h.product() << " " << h.provenance() << " " << h.id();
    return os;
  }

  template <typename ELEMENT>
  bool
  DataViewImpl::getByLabel(std::string const& moduleLabel,
			   std::string const& productInstanceName,
			   Handle<View<ELEMENT> >& result) const
  {
   result.clear();

    TypeID typeID(typeid(ELEMENT));

    BasicHandleVec bhv;
    int nFound = getMatchingSequenceByLabel_(typeID,
                                             moduleLabel,
                                             productInstanceName,
                                             bhv,
                                             true);

    if (nFound == 0) {
      boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound) );
      *whyFailed
	<< "getByLabel: Found zero products matching all criteria\n"
	<< "Looking for sequence of type: " << typeID << "\n"
	<< "Looking for module label: " << moduleLabel << "\n"
	<< "Looking for productInstanceName: " << productInstanceName << "\n";
      Handle<View<ELEMENT> > temp(whyFailed);
      result.swap(temp);
      return false;
    }
    if (nFound > 1) {
      throw edm::Exception(edm::errors::ProductNotFound)
        << "getByLabel: Found more than one product matching all criteria\n"
	<< "Looking for sequence of type: " << typeID << "\n"
	<< "Looking for module label: " << moduleLabel << "\n"
	<< "Looking for productInstanceName: " << productInstanceName << "\n";
    }

    fillView_(bhv[0], result);
    return true;
  }

  template <typename ELEMENT>
    bool
    DataViewImpl::getByLabel(InputTag const& tag, Handle<View<ELEMENT> >& result) const
  {
    result.clear();
    if (tag.process().empty()) {
      return getByLabel(tag.label(), tag.instance(), result);
    } else {
      TypeID typeID(typeid(ELEMENT));
      
      BasicHandleVec bhv;
      int nFound = getMatchingSequenceByLabel_(typeID,
                                               tag.label(),
                                               tag.instance(),
                                               tag.process(),
                                               bhv,
                                               true);
      
      if (nFound == 0) {
        boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound) );
        *whyFailed
          << "getByLabel: Found zero products matching all criteria\n"
          << "Looking for sequence of type: " << typeID << "\n"
          << "Looking for module label: " << tag.label() << "\n"
          << "Looking for productInstanceName: " << tag.instance() << "\n"
          << "Looking for processName: "<<tag.process() <<"\n";
        Handle<View<ELEMENT> > temp(whyFailed);
        result.swap(temp);
        return false;
      }
      if (nFound > 1) {
        throw edm::Exception(edm::errors::ProductNotFound)
        << "getByLabel: Found more than one product matching all criteria\n"
	<< "Looking for sequence of type: " << typeID << "\n"
        << "Looking for module label: " << tag.label() << "\n"
        << "Looking for productInstanceName: " << tag.instance() << "\n"
        << "Looking for processName: "<<tag.process() <<"\n";
      }
      
      fillView_(bhv[0], result);
      return true;
    }
    return false;
  }

    template <typename ELEMENT>
    bool
    DataViewImpl::get(ProductID const& oid, Handle<View<ELEMENT> >& result) const
  {
      result.clear();
      BasicHandle bh = this->get_(oid);

      if(bh.failedToGet()) {
          boost::shared_ptr<cms::Exception> whyFailed(new edm::Exception(edm::errors::ProductNotFound) );
          *whyFailed
              << "get View by ID failed: no product with ID = " << oid.id() <<"\n";
          Handle<View<ELEMENT> > temp(whyFailed);
          result.swap(temp);
          return false;
      }

      fillView_(bh, result);
      return true;
  }
  
  
  template <typename ELEMENT>
  void
  DataViewImpl::fillView_(BasicHandle & bh,
			  Handle<View<ELEMENT> >& result) const
  {
    std::vector<void const*> pointersToElements;
    // the following is a shared pointer. 
    // It is not initialized here
    helper_vector_ptr helpers;
    // the following must initialize the
    //  shared pointer and fill the helper vector
    bh.wrapper()->fillView(bh.id(), pointersToElements, helpers);

    boost::shared_ptr<View<ELEMENT> > 
      newview(new View<ELEMENT>(pointersToElements, helpers));
    
    gotProductIDs_.insert(bh.id());
    gotViews_.push_back(newview);
    Handle<View<ELEMENT> > h(&*newview, bh.provenance());
    result.swap(h);
  }

  template <typename PROD>
  void 
  DataViewImpl::getMany(SelectorBase const& sel,
			std::vector<Handle<PROD> >& results) const
  { 
    BasicHandleVec bhv;
    this->getMany_(TypeID(typeid(PROD)), sel, bhv);
    
    // Go through the returned handles; for each element,
    //   1. create a Handle<PROD> and
    //   2. record the ProductID in gotProductIDs
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

    BasicHandleVec::const_iterator it = bhv.begin();
    BasicHandleVec::const_iterator end = bhv.end();

    while (it != end) {
      gotProductIDs_.insert((*it).id());
      Handle<PROD> result;
      convert_handle(*it, result);  // throws on conversion error
      products.push_back(result);
      ++it;
    }
    results.swap(products);
  }

  template <typename PROD>
  bool
  DataViewImpl::getByType(Handle<PROD>& result) const
  {
    result.clear();
    BasicHandle bh = this->getByType_(TypeID(typeid(PROD)));
    convert_handle(bh, result);  // throws on conversion error
    if(!bh.failedToGet()) {
      gotProductIDs_.insert(bh.id());
      return true;
    }
    return false;
  }

  template <typename PROD>
  void 
  DataViewImpl::getManyByType(std::vector<Handle<PROD> >& results) const
  { 
    BasicHandleVec bhv;
    this->getManyByType_(TypeID(typeid(PROD)), bhv);
    
    // Go through the returned handles; for each element,
    //   1. create a Handle<PROD> and
    //   2. record the ProductID in gotProductIDs
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

    BasicHandleVec::const_iterator it = bhv.begin();
    BasicHandleVec::const_iterator end = bhv.end();

    while (it != end) {
      gotProductIDs_.insert((*it).id());
      Handle<PROD> result;
      convert_handle(*it, result);  // throws on conversion error
      products.push_back(result);
      ++it;
    }
    results.swap(products);
  }
}
#endif
