#ifndef Framework_Event_h
#define Framework_Event_h

/*----------------------------------------------------------------------

Event: This is the primary interface for accessing
EDProducts from a single collision and inserting new derived products.

$Id: Event.h,v 1.21 2005/09/30 21:16:00 paterno Exp $

----------------------------------------------------------------------*/
#include <cassert>
#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

#include "FWCore/EDProduct/interface/EventID.h"
#include "FWCore/EDProduct/interface/Timestamp.h"
#include "FWCore/EDProduct/interface/traits.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/BasicHandle.h"


#include "FWCore/Framework/src/Group.h"
#include "FWCore/Framework/src/TypeID.h"

namespace edm {

  class Event
  {
  public:
    Event(EventPrincipal& ep, const ModuleDescription&);
    ~Event();

    // these must come from the ep.
    EventID id() const;
    Timestamp time() const;
    
    // How do these get set in the first place?
    const LuminositySection& getLuminositySection() const;
    const Run& getRun() const;

    template <class PROD>
    void 
    put(std::auto_ptr<PROD> product);

    template <class PROD>
    void 
    put(std::auto_ptr<PROD> product, const std::string& productInstanceName);

    template <class PROD>
    void 
    get(ProductID const& id, Handle<PROD>& result) const;

    template <class PROD>
    void 
    get(Selector const&, Handle<PROD>& result) const;
  
    template <class PROD>
    void 
    getByLabel(std::string const& label, Handle<PROD>& result) const;

    template <class PROD>
    void 
    getByLabel(std::string const& label, const std::string& productInstanceName, Handle<PROD>& result) const;

    template <class PROD>
    void 
    getMany(Selector const&, std::vector<Handle<PROD> >& results) const;

    template <class PROD>
    void
    getByType(Handle<PROD>& result) const;

    template <class PROD>
    void 
    getManyByType(std::vector<Handle<PROD> >& results) const;

    Provenance const&
    getProvenance(ProductID const& id) const;

    void
    getAllProvenance(std::vector<Provenance const*> &provenances) const;

  private:
    typedef std::vector<ProductID>       ProductIDVec;
    //typedef std::vector<const Group*> GroupPtrVec;
    typedef std::vector<std::pair<EDProduct*, std::string> >   ProductPtrVec;
    typedef std::vector<BasicHandle>  BasicHandleVec;    

    //------------------------------------------------------------
    // Private functions.
    //
    // commit() is called to complete the transaction represented by
    // this Event. The friendship required seems gross, but any
    // alternative is not great either.  and putting it into the
    // public interface is asking for trouble
    void commit_();
    friend class ProducerWorker;

    // The following 'get' functions serve to isolate the Event class
    // from the EventPrincipal class.

    BasicHandle 
    get_(ProductID const& oid) const;

    BasicHandle 
    get_(TypeID const& id, Selector const&) const;
    
    BasicHandle 
    getByLabel_(TypeID const& id,
		std::string const& label,
		const std::string& productInstanceName) const;

    void 
    getMany_(TypeID const& id, 
	     Selector const& sel, 
	     BasicHandleVec& results) const;

    BasicHandle 
    getByType_(TypeID const& id) const;

    void 
    getManyByType_(TypeID const& id, 
	     BasicHandleVec& results) const;

    //------------------------------------------------------------
    // Copying and assignment of Events is disallowed
    //
    Event(const Event&);                  // not implemented
    const void operator=(const Event&);   // not implemented

    //------------------------------------------------------------
    // Data members
    //

    // put_products_ is the holding pen for EDProducts inserted into
    // this Event. Pointers in this collection own the products to
    // which they point.
    ProductPtrVec            put_products_;

    // gotProductIDs_ must be mutable because it records all 'gets',
    // which do not logically modify the Event. gotProductIDs_ is
    // merely a cache reflecting what has been retreived from the
    // EventPrincipal.
    mutable ProductIDVec        gotProductIDs_;

    // Each Event must have an associated EventPrincipal, used as the
    // source of all 'gets' and the target of 'puts'.
    EventPrincipal&          ep_;

    // Each Event must have a description of the module executing the
    // "transaction" which the Event represents.
    const ModuleDescription& md_;
  };


  //------------------------------------------------------------
  //
  // Utilities for creation of handles.
  //
  
  template <class PROD>
  Handle<PROD> make_handle(const Group& g)
  {
    return Handle<PROD>(g.product(), g.provenance());
  }
 
  template <class PROD>
  struct makeHandle
  {
    Handle<PROD> operator()(const Group& g) { return make_handle<PROD>(g); }
    Handle<PROD> operator()(const Group* g) { return make_handle<PROD>(*g); }
  };


  //------------------------------------------------------------
  // Metafunction support for compile-time selection of code used in
  // Event::put member template.
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
#if 0
    typedef char (& no_tag )[1]; // type indicating FALSE
    typedef char (& yes_tag)[2]; // type indicating TRUE

    // Definitions forthe following struct and function templates are
    // not needed; we only require the declarations.
    template <typename T, void (T::*)()>  struct ptmf_helper;
    template <typename T> no_tag  has_postinsert_helper(...);
    template <typename T> yes_tag has_postinsert_helper(ptmf_helper<T, &T::post_insert> * p);

    template< typename T >
    struct has_postinsert
    {
      static bool const value = 
	sizeof(has_postinsert_helper<T>(0)) == sizeof(yes_tag);
    };
#else
    //------------------------------------------------------------
    // THE FOLLOWING SHOULD BE REMOVED when we move to a newer
    // compiler; see the note above.
    //------------------------------------------------------------

    //------------------------------------------------------------
    // The definition of the primary template should be in its own
    // header, in EDProduct; this is because anyone who specializes
    // this template has to include the declaration of the primary
    // template.
    //------------------------------------------------------------


    template< typename T >
    struct has_postinsert
    {
      static bool const value = has_postinsert_trait<T>::value;	
    };

#endif
  }



  //------------------------------------------------------------

  // The following function objects are used by Event::put, under the
  // control of a metafunction if, to either call the given object's
  // post_insert function (if it has one), or to do nothing (if it
  // does not have a post_insert function).
  template <class T>
  struct DoPostInsert
  {
    void operator()(T* p) const { p->post_insert(); }
  };

  template <class T>
  struct DoNothing
  {
    void operator()(T*) const { }
  };


  //------------------------------------------------------------
  //
  // Implementation of  Event  member templates. See  Event.cc for the
  // implementation of non-template members.
  //

  template <class PROD>
  inline
  void 
  Event::put(std::auto_ptr<PROD> product)
  {
    put(product, std::string());
  }

  template <class PROD>
  void 
  Event::put(std::auto_ptr<PROD> product, const std::string& productInstanceName)
  {
    PROD* p = product.get();
    assert (p);                // null pointer is illegal

    // The following will call post_insert if T has such a function,
    // and do nothing if T has no such function.
    typename boost::mpl::if_c<detail::has_postinsert<PROD>::value, 
                              DoPostInsert<PROD>, 
                              DoNothing<PROD> >::type maybe_inserter;
    maybe_inserter(p);

    edm::Wrapper<PROD> *wp(new Wrapper<PROD>(*p));
    put_products_.push_back(std::make_pair(wp, productInstanceName));
    // product.release(); // The object has been copied into the Wrapper.
    // The old copy must be deleted, so we cannot release ownership.
  }

  template <class PROD>
  void
  Event::get(ProductID const& id, Handle<PROD>& result) const
  {
    BasicHandle bh = this->get_(TypeID(typeid(PROD)), id);
    gotProductIDs_.push_back(bh.id());
    convert_handle(bh, result);  // throws on conversion error
  }

  template <class PROD>
  void 
  Event::get(Selector const& sel,
	     Handle<PROD>& result) const
  {
    BasicHandle bh = this->get_(TypeID(typeid(PROD)),sel);
    gotProductIDs_.push_back(bh.id());
    convert_handle(bh, result);  // throws on conversion error
  }
  
  template <class PROD>
  inline
  void
  Event::getByLabel(std::string const& label,
		    Handle<PROD>& result) const
  {
    getByLabel(label, std::string(), result);
  }

  template <class PROD>
  void
  Event::getByLabel(std::string const& label,
                    const std::string& productInstanceName,
		    Handle<PROD>& result) const
  {
    BasicHandle bh = this->getByLabel_(TypeID(typeid(PROD)), label, productInstanceName);
    gotProductIDs_.push_back(bh.id());
    convert_handle(bh, result);  // throws on conversion error
  }

  template <class PROD>
  void 
  Event::getMany(Selector const& sel,
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
	gotProductIDs_.push_back((*it).id());
	Handle<PROD> result;
	convert_handle(*it, result);  // throws on conversion error
	products.push_back(result);
	++it;
    }
    results.swap(products);
  }

  template <class PROD>
  void
  Event::getByType(Handle<PROD>& result) const
  {
    BasicHandle bh = this->getByType_(TypeID(typeid(PROD)));
    gotProductIDs_.push_back(bh.id());
    convert_handle(bh, result);  // throws on conversion error
  }

  template <class PROD>
  void 
  Event::getManyByType(std::vector<Handle<PROD> >& results) const
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
	gotProductIDs_.push_back((*it).id());
	Handle<PROD> result;
	convert_handle(*it, result);  // throws on conversion error
	products.push_back(result);
	++it;
    }
    results.swap(products);
  }

}
#endif
