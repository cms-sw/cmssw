#ifndef FWCore_Framework_PrincipalGetAdapter_h
#define FWCore_Framework_PrincipalGetAdapter_h

// -*- C++ -*-
//

// Class  :     PrincipalGetAdapter
// 
/**\class PrincipalGetAdapter PrincipalGetAdapter.h FWCore/Framework/interface/PrincipalGetAdapter.h

Description: This is the implementation for accessing EDProducts and 
inserting new EDproducts.

Usage:

Getting Data

The edm::PrincipalGetAdapter class provides many 'get*" methods for getting data
it contains.  

The primary method for getting data is to use getByLabel(). The labels are
the label of the module assigned in the configuration file and the 'product
instance label' (which can be omitted in the case the 'product instance label'
is the default value).  The C++ type of the product plus the two labels
uniquely identify a product in the PrincipalGetAdapter.

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
std::auto_ptr<AppleCollection> pApples(new AppleCollection);
  
//fill the collection
...
event.put(pApples);
\endcode

\code
std::auto_ptr<FruitCollection> pFruits(new FruitCollection);

//fill the collection
...
event.put("apple", pFruits);
\endcode


Getting a reference to a product before that product is put into the
event/lumiBlock/run.
NOTE: The edm::RefProd returned will not work until after the
edm::PrincipalGetAdapter has been committed (which happens after the
EDProducer::produce method has ended)
\code
std::auto_ptr<AppleCollection> pApples(new AppleCollection);

edm::RefProd<AppleCollection> refApples = event.getRefBeforePut<AppleCollection>();

//do loop and fill collection
for(unsigned int index = 0; .....) {
....
apples->push_back(Apple(...));
  
//create an edm::Ref to the new object
edm::Ref<AppleCollection> ref(refApples, index);
....
}
\endcode

*/
/*----------------------------------------------------------------------

----------------------------------------------------------------------*/
#include <cassert>
#include <typeinfo>
#include <string>
#include <vector>

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Common/interface/traits.h"

#include "DataFormats/Common/interface/BasicHandle.h"

#include "DataFormats/Common/interface/ConvertHandle.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Wrapper.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"


namespace edm {

  class ModuleCallingContext;

  namespace principal_get_adapter_detail {
    struct deleter {
      void operator()(std::pair<WrapperOwningHolder, BranchDescription const*> const p) const;
    };
    void
    throwOnPutOfNullProduct(char const* principalType, TypeID const& productType, std::string const& productInstanceName);
    void
    throwOnPrematureRead(char const* principalType, TypeID const& productType, std::string const& moduleLabel, std::string const& productInstanceName);
    void
    throwOnPrematureRead(char const* principalType, TypeID const& productType);

    void
    throwOnPrematureRead(char const* principalType, TypeID const& productType, EDGetToken);

  }
  class PrincipalGetAdapter {
  public:
    PrincipalGetAdapter(Principal & pcpl,
		 ModuleDescription const& md);

    ~PrincipalGetAdapter();

    PrincipalGetAdapter(PrincipalGetAdapter const&) = delete; // Disallow copying and moving
    PrincipalGetAdapter& operator=(PrincipalGetAdapter const&) = delete; // Disallow copying and moving

    //size_t size() const;
    
    void setConsumer(EDConsumerBase const* iConsumer) {
      consumer_ = iConsumer;
    }

    bool isComplete() const;

    template <typename PROD>
    bool 
    checkIfComplete() const;

    template <typename PROD>
    void 
    getManyByType(std::vector<Handle<PROD> >& results, ModuleCallingContext const* mcc) const;

    ProcessHistory const&
    processHistory() const;

    Principal& principal() {return principal_;}
    Principal const& principal() const {return principal_;}

    BranchDescription const&
    getBranchDescription(TypeID const& type, std::string const& productInstanceName) const;

    typedef std::vector<BasicHandle>  BasicHandleVec;

    //------------------------------------------------------------
    // Protected functions.
    //

    // The following 'get' functions serve to isolate the PrincipalGetAdapter class
    // from the Principal class.

    BasicHandle 
    getByLabel_(TypeID const& tid, InputTag const& tag,
                ModuleCallingContext const* mcc) const;

    BasicHandle 
    getByLabel_(TypeID const& tid,
                std::string const& label,
                std::string const& instance,
                std::string const& process,
                ModuleCallingContext const* mcc) const;

    BasicHandle
    getByToken_(TypeID const& id, KindOfType kindOfType, EDGetToken token,
                ModuleCallingContext const* mcc) const;

    BasicHandle
    getMatchingSequenceByLabel_(TypeID const& typeID,
                                InputTag const& tag,
                                ModuleCallingContext const* mcc) const;

    BasicHandle
    getMatchingSequenceByLabel_(TypeID const& typeID,
                                std::string const& label,
                                std::string const& instance,
                                std::string const& process,
                                ModuleCallingContext const* mcc) const;
    
    void 
    getManyByType_(TypeID const& tid, 
		   BasicHandleVec& results,
                   ModuleCallingContext const* mcc) const;

    // Also isolates the PrincipalGetAdapter class
    // from the Principal class.
    EDProductGetter const* prodGetter() const;

  private:
    // Is this an Event, a LuminosityBlock, or a Run.
    BranchType const& branchType() const;

    BasicHandle
    makeFailToGetException(KindOfType,TypeID const&,EDGetToken) const;

    void
    throwAmbiguousException(TypeID const& productType, EDGetToken token) const;

  private:
    //------------------------------------------------------------
    // Data members
    //

    // Each PrincipalGetAdapter must have an associated Principal, used as the
    // source of all 'gets' and the target of 'puts'.
    Principal & principal_;

    // Each PrincipalGetAdapter must have a description of the module executing the
    // "transaction" which the PrincipalGetAdapter represents.
    ModuleDescription const& md_;
    
    EDConsumerBase const* consumer_;

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
  // PrincipalGetAdapter::put member template.
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
    typedef char (& no_tag)[1]; // type indicating FALSE
    typedef char (& yes_tag)[2]; // type indicating TRUE

    // Definitions forthe following struct and function templates are
    // not needed; we only require the declarations.
    template <typename T, void (T::*)()>  struct postinsert_function;
    template <typename T> no_tag  has_postinsert_helper(...);
    template <typename T> yes_tag has_postinsert_helper(postinsert_function<T, &T::post_insert> * p);


    template<typename T>
    struct has_postinsert {
      static bool const value = 
	sizeof(has_postinsert_helper<T>(nullptr)) == sizeof(yes_tag) &&
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

  // Implementation of  PrincipalGetAdapter  member templates. See  PrincipalGetAdapter.cc for the
  // implementation of non-template members.
  //

  template <typename PROD>
  inline
  bool 
  PrincipalGetAdapter::checkIfComplete() const { 
    return isComplete() || !detail::has_mergeProduct_function<PROD>::value;
  }

  template <typename PROD>
  inline
  void 
  PrincipalGetAdapter::getManyByType(std::vector<Handle<PROD> >& results,
                                     ModuleCallingContext const* mcc) const { 
    BasicHandleVec bhv;
    this->getManyByType_(TypeID(typeid(PROD)), bhv, mcc);
    
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

    typename BasicHandleVec::iterator it = bhv.begin();
    typename BasicHandleVec::iterator end = bhv.end();

    while (it != end) {
      Handle<PROD> result;
      convert_handle(std::move(*it), result);  // throws on conversion error
      products.push_back(result);
      ++it;
    }
    results.swap(products);
  }
}
#endif
