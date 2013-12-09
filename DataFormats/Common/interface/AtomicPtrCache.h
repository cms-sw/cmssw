#ifndef DataFormats_Common_AtomicPtrCache_h
#define DataFormats_Common_AtomicPtrCache_h
// -*- C++ -*-
//
// Package:     DataFormats/Common
// Class  :     AtomicPtrCache
// 
/**\class edm::AtomicPtrCache AtomicPtrCache.h "DataFormats/Common/interface/AtomicPtrCache.h"

 Description: A thread safe cache managed by a pointer

 Usage:
    Data products which need to cache results into non-trivial structures (e.g. an std::vector)
 can use this class to manage the cache in a thread-safe way.
 The thread-safety guarantee is only the standard C++, if calls are made to const functions simultaneously
 then everything is thread safe. Calling a non-const function while calling any other functions
 is not thread-safe.
 
 This class also hides the std::atomic from ROOT so this class can safely be used in a stored class.

 WARNING: member data which uses this class must be made transient in the classes_def.xml file!

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 07 Nov 2013 00:50:40 GMT
//

// system include files
 #if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#include <memory>
#endif
// user include files

// forward declarations

namespace edm {
  template <typename T>
  class AtomicPtrCache
  {
    
  public:
    AtomicPtrCache();
    
    ///Takes exclusive ownership of the value
    explicit AtomicPtrCache(T*);
    
    ///Uses T's copy constructor to make a copy
    AtomicPtrCache(const AtomicPtrCache<T>&);
    AtomicPtrCache& operator=(const AtomicPtrCache<T>&);

    ~AtomicPtrCache();
    
    // ---------- const member functions ---------------------
    T const* operator->() const { return load();}
    T const& operator*() const {return *load(); }
    T const* load() const;

    bool isSet() const;
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    ///returns true if actually was set.
    /// Will delete value held by iNewValue if not the first time set
    bool set(std::unique_ptr<T> iNewValue) const;
#endif
    // ---------- static member functions --------------------
    
    // ---------- member functions ---------------------------
    T* operator->() { return load();}
    T& operator*() {return *load();}
    
    T* load();
    
    ///unsets the value and deletes the memory
    void reset();
    
  private:
    
    // ---------- member data --------------------------------
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    mutable std::atomic<T*> m_data;
#else
    mutable T* m_data;
#endif
  };
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  template<typename T>
  inline AtomicPtrCache<T>::AtomicPtrCache():m_data{nullptr} {}
  
  template<typename T>
  inline AtomicPtrCache<T>::AtomicPtrCache(T* iValue): m_data{iValue} {}
  
  template<typename T>
  inline AtomicPtrCache<T>::AtomicPtrCache(const AtomicPtrCache<T>& iOther):
  m_data{nullptr}
  {
    auto ptr = iOther.m_data.load(std::memory_order_acquire);
    if(ptr != nullptr) {
      m_data.store( new T{*ptr}, std::memory_order_release);
    }
  }
  template<typename T>
  inline AtomicPtrCache<T>& AtomicPtrCache<T>::operator=(const AtomicPtrCache<T>& iOther) {
    auto ptr = iOther.m_data.load(std::memory_order_acquire);
    if(ptr != nullptr) {
      auto ourPtr =m_data.load(std::memory_order_acquire);
      if( ourPtr !=nullptr) {
        *ourPtr = *ptr;
      } else {
        m_data.store( new T{*ptr}, std::memory_order_release);
      }
    } else {
      delete m_data.exchange(nullptr, std::memory_order_acq_rel);
    }
    return *this;
  }

  
  template<typename T>
  inline AtomicPtrCache<T>::~AtomicPtrCache(){
    delete m_data.load(std::memory_order_acquire);
  }
  
  template<typename T>
  inline T* AtomicPtrCache<T>::load(){ return m_data.load(std::memory_order_acquire);}

  template<typename T>
  inline T const* AtomicPtrCache<T>::load() const{ return m_data.load(std::memory_order_acquire);}
  
  template<typename T>
  inline bool AtomicPtrCache<T>::isSet() const { return nullptr!=m_data.load(std::memory_order_acquire);}

  template<typename T>
  inline bool AtomicPtrCache<T>::set(std::unique_ptr<T> iNewValue) const {
    bool retValue;
    T* expected = nullptr;
    if( (retValue = m_data.compare_exchange_strong(expected,iNewValue.get(), std::memory_order_acq_rel)) ) {
      iNewValue.release();
    }
    return retValue;
  }

  template<typename T>
  inline void AtomicPtrCache<T>::reset() { delete m_data.exchange(nullptr,std::memory_order_acq_rel);}

#endif
}


#endif
