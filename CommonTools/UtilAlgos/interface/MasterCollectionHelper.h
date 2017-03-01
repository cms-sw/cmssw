#ifndef CommonTools_UtilAlgos_MasterCollectionHelper_h
#define CommonTools_UtilAlgos_MasterCollectionHelper_h
/* \class helper::MasterCollection<C>
 *
 * differentiate index addressing in case of edm::View
 * since a View retrieving a collection of reference
 * would have referece indices (key()) pointing to the master
 * collection different from reference position into the view
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: MasterCollectionHelper.h,v 1.1 2009/03/03 13:07:26 llista Exp $
 *
 */
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

namespace helper {
  template<typename C1>
  struct MasterCollection {
    typedef edm::Ref<C1> ref_type;
    explicit MasterCollection(const edm::Handle<C1> & handle, edm::Event const& event) : 
      handle_(handle) { }
    size_t size() const { return handle_->size(); }
    size_t index(size_t i) const { return i; }
    const edm::Handle<C1> & get() const { return handle_; }
    ref_type getRef(size_t idx) const { return ref_type(get(), idx); }
    template<typename R>
    R getConcreteRef(size_t idx) const { return getRef(idx); }
  private:
    edm::Handle<C1> handle_;
  };
  
  template<typename T>
  struct MasterCollection<edm::View<T> > {
    typedef edm::RefToBase<T> ref_type;
    explicit MasterCollection(const edm::Handle<edm::View<T> > & handle, edm::Event const& event) :
      handle_(handle) {
      if(handle_->size() != 0) {
        ref_ = edm::makeRefToBaseProdFrom(handle_->refAt(0), event);
      }
    }
    size_t size() const { 
      if (ref_.isNull()) return 0;
      return ref_->size(); 
    }
    size_t index(size_t i) const { 
      return handle_->refAt(i).key(); 
    }
    const edm::RefToBaseProd<T> & get() const { return ref_; }
    ref_type getRef(size_t idx) const { return ref_type(get(), idx); }
    template<typename R>
    R getConcreteRef(size_t idx) const { return getRef(idx).template castTo<R>(); }
  private:
    edm::Handle<edm::View<T> > handle_;
    edm::RefToBaseProd<T> ref_;
  };

}

#endif

