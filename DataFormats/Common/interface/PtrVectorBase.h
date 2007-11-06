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
// $Id$
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
    
    /// True if the data has already be obtained from the Event
    bool hasCache() const { return 0!=core_.productPtr(); }
    
    /// Is the RefVector empty
    bool empty() const {return indicies_.empty();}
    
    /// Size of the RefVector
    size_type size() const {return indicies_.size();}
    
    /// Capacity of the RefVector
    size_type capacity() const {return indicies_.capacity();}
    
    bool operator==(const PtrVectorBase& iRHS) const;
    // ---------- static member functions --------------------
    
    // ---------- member functions ---------------------------
    /// Reserve space for RefVector
    void reserve(size_type n) {indicies_.reserve(n); cachedItems_.reserve(n);}
    
    void setProductGetter(EDProductGetter* iGetter) const { core_.setProductGetter(iGetter); }
  protected:
    PtrVectorBase();
    void push_back_base(key_type iKey, 
                        const void* iData,
                        const ProductID& iID,
                        const EDProductGetter* iGetter);
    
    std::vector<const void*>::const_iterator void_begin() const {
      getProduct_();
      return cachedItems_.begin();
    }
    std::vector<const void*>::const_iterator void_end() const {
      getProduct_();
      return cachedItems_.end();
    }
    
    template< class TPtr>
    TPtr makePtr(unsigned long iIndex) const {
      getProduct_();
      return TPtr(this->id(),
                  reinterpret_cast<const typename TPtr::value_type*>(cachedItems_[iIndex]),
                  iIndex);
    }
    
    template< class TPtr>
    TPtr makePtr(const std::vector<const void*>::const_iterator iIt) const {
      getProduct_();
      return TPtr(this->id(),
                  reinterpret_cast<const typename TPtr::value_type*>(*iIt),
                  iIt - cachedItems_.begin());
    }
    
  private:
    //PtrVectorBase(const PtrVectorBase&); // stop default
    
    //const PtrVectorBase& operator=(const PtrVectorBase&); // stop default
    void getProduct_() const;
    virtual const std::type_info& typeInfo() const=0;
    // ---------- member data --------------------------------
    //NOTE: the productPtr of the RefCore is used to determine if the data has yet been retrieved
    RefCore core_;
    std::vector<key_type> indicies_;
    mutable std::vector<const void*> cachedItems_;
  };
  
}

#endif

