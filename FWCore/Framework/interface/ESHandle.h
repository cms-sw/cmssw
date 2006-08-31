#ifndef Framework_ESHandle_h
#define Framework_ESHandle_h
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
// $Id: ESHandle.h,v 1.9 2006/08/26 18:39:49 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ComponentDescription.h"

// forward declarations
namespace edm {
template< class T>
class ESHandle
{

   public:
      typedef T value_type;
   
      ESHandle() : data_(0), description_(0) {}
      ESHandle(const T* iData) : data_(iData), description_(0) {}
//      { std::cout<<"Call ESHanlde(data) ctor"<<std::endl; }
      ESHandle(const T* iData, const edm::eventsetup::ComponentDescription* desc) 
           : data_(iData), description_(desc) {}
//      { std::cout<<"Call ESHanlde(data,desc) ctor"<<std::endl; }
      //virtual ~ESHandle();

      // ---------- const member functions ---------------------
      const T* product() const { return data_; }
      const T* operator->() const { return product(); }
      const T& operator*() const { return *product(); }
      const edm::eventsetup::ComponentDescription* description() const { 
         if(!description_) {
            throw edm::Exception(edm::errors::InvalidReference,"NullPointer");
         }
         return description_; 
      }
      
      bool isValid() const { return 0 != data_ && 0 != description_; }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void swap(ESHandle<T>& iOther) {
         std::swap(data_, iOther.data_);
         std::swap(description_, iOther.description_);
      }
      
   private:
      //ESHandle(const ESHandle&); // stop default

      //const ESHandle& operator=(const ESHandle&); // stop default

      // ---------- member data --------------------------------
      const T* data_; 
      const edm::eventsetup::ComponentDescription* description_;
};

  // Free swap function
  template <class T>
  inline
  void
  swap(ESHandle<T>& a, ESHandle<T>& b) 
  {
    a.swap(b);
  }
}
#endif
