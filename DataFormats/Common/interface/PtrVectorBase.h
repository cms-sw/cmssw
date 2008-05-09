#ifndef DataFormats_Common_PtrVectorBase_h
#define DataFormats_Common_PtrVectorBase_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     PtrVectorBase
// 
/**\class PtrVectorBase PtrVectorBase.h DataFormats/Common/interface/PtrVectorBase.h

 Description: Base class for PtrVector

 Usage:
    This class defines the common behavior for the PtrVector template class instances

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Oct 24 15:26:45 EDT 2007
// $Id: PtrVectorBase.h,v 1.6 2008/02/16 17:25:48 wmtan Exp $
//

// system include files
#include <vector>

// user include files
#include "DataFormats/Common/interface/RefCore.h"

// forward declarations

namespace edm {
  class PtrVectorBase {
    
  public:
    typedef unsigned long key_type;
    typedef key_type size_type;
    
    virtual ~PtrVectorBase();
    
    // ---------- const member functions ---------------------
    /// Checks for null
    bool isNull() const {return !isNonnull(); }
    
    /// Checks for non-null
    //bool isNonnull() const {return id().isValid(); }
    bool isNonnull() const { return core_.isNonnull(); }
    
    /// Checks for null
    bool operator!() const {return isNull();}
    
    /// Accessor for product ID.
    ProductID id() const {return core_.id();}
    
    /// Accessor for product getter.
    EDProductGetter const* productGetter() const {return core_.productGetter();}
    
    bool hasCache() const { return !cachedItems_.empty(); }
    
    /// True if the data is in memory or is available in the Event
    /// No type checking is done.
    bool isAvailable() const{ return core_.isAvailable(); }

    /// Is the RefVector empty
    bool empty() const {return indicies_.empty();}
    
    /// Size of the RefVector
    size_type size() const {return indicies_.size();}
    
    /// Capacity of the RefVector
    size_type capacity() const {return indicies_.capacity();}

    /// Clear the PtrVector
    void clear() { core_ = RefCore(); indicies_.clear(); cachedItems_.clear(); }
    
    bool operator==(PtrVectorBase const& iRHS) const;
    // ---------- static member functions --------------------
    
    // ---------- member functions ---------------------------
    /// Reserve space for RefVector
    void reserve(size_type n) {indicies_.reserve(n); cachedItems_.reserve(n);}
    
    void setProductGetter(EDProductGetter* iGetter) const { core_.setProductGetter(iGetter); }

    bool isTransient() const {return core_.isTransient();}

  protected:
    PtrVectorBase();

    /// swap
    void swap(PtrVectorBase& other);

    void push_back_base(RefCore const& core, key_type iKey, void const* iData);
    
    std::vector<void const*>::const_iterator void_begin() const {
      getProduct_();
      return cachedItems_.begin();
    }
    std::vector<void const*>::const_iterator void_end() const {
      getProduct_();
      return cachedItems_.end();
    }
    
    template<typename TPtr>
    TPtr makePtr(unsigned long iIndex) const {
      if (isTransient()) {
        return TPtr(reinterpret_cast<typename TPtr::value_type const*>(cachedItems_[iIndex]),
                  indicies_[iIndex]);
      }
      if (hasCache()) {
        return TPtr(this->id(),
                  reinterpret_cast<typename TPtr::value_type const*>(cachedItems_[iIndex]),
                  indicies_[iIndex]);
      }
      return TPtr(this->id(), indicies_[iIndex], productGetter());
    }
    
    template<typename TPtr>
    TPtr makePtr(std::vector<void const*>::const_iterator const iIt) const {
      if (isTransient()) {
        return TPtr(reinterpret_cast<typename TPtr::value_type const*>(*iIt),
                  indicies_[iIt - cachedItems_.begin()]);
      }
      if (hasCache()) {
        return TPtr(this->id(),
                  reinterpret_cast<typename TPtr::value_type const*>(*iIt),
                  indicies_[iIt - cachedItems_.begin()]);
      }
      return TPtr(this->id(), indicies_[iIt - cachedItems_.begin()], productGetter());
    }
    
  private:
    void getProduct_() const;
    virtual std::type_info const& typeInfo() const=0;
    // ---------- member data --------------------------------
    RefCore core_;
    std::vector<key_type> indicies_;
    mutable std::vector<void const*> cachedItems_; //! transient
  };
}

#endif
