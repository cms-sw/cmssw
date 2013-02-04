#ifndef FWCore_Framework_ESHandle_h
#define FWCore_Framework_ESHandle_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESHandle
// 
/**\class ESHandle ESHandle.h FWCore/Framework/interface/ESHandle.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  1 14:47:35 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/ComponentDescription.h"

// forward declarations
namespace edm {

class ESHandleBase {
   public:
      ESHandleBase() : data_(nullptr), description_(nullptr) {}
      ESHandleBase(void const* iData, edm::eventsetup::ComponentDescription const* desc) 
           : data_(iData), description_(desc) {}

      edm::eventsetup::ComponentDescription const* description() const;
      
      bool isValid() const { return 0 != data_ && 0 != description_; }

      void swap(ESHandleBase& iOther) {
         std::swap(data_, iOther.data_);
         std::swap(description_, iOther.description_);
      }
   protected:
     void const *productStorage() const {return data_;}

   private:
      // ---------- member data --------------------------------
      void const* data_; 
      edm::eventsetup::ComponentDescription const* description_;
};

template<typename T>
class ESHandle : public ESHandleBase {
   public:
      typedef T value_type;
   
      ESHandle() : ESHandleBase() {}
      ESHandle(T const* iData) : ESHandleBase(iData, 0) {}
      ESHandle(T const* iData, edm::eventsetup::ComponentDescription const* desc) : ESHandleBase(iData, desc) {}

      // ---------- const member functions ---------------------
      T const* product() const { return static_cast<T const *>(productStorage()); }
      T const* operator->() const { return product(); }
      T const& operator*() const { return *product(); }
      // ---------- static member functions --------------------
      static const bool transientAccessOnly = false;

      // ---------- member functions ---------------------------
      
   private:
};

  // Free swap function
  inline
  void
  swap(ESHandleBase& a, ESHandleBase& b) 
  {
    a.swap(b);
  }
}
#endif
