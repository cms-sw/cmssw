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
// $Id: ESHandle.h,v 1.3 2005/07/14 22:50:52 wmtan Exp $
//

// system include files

// user include files

// forward declarations
namespace edm {
   namespace eventsetup {
template< class T>
class ESHandle
{

   public:
      typedef T value_type;
   
      ESHandle() : data_(0) {}
      ESHandle(const T* iData) : data_(iData) {}
      //virtual ~ESHandle();

      // ---------- const member functions ---------------------
      const T* product() const { return data_; }
      const T* operator->() const { return product(); }
      const T& operator*() const { return *product(); }
      
      bool isValid() const { return 0 != data_; }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void swap(ESHandle<T>& iOther) {
         std::swap(data_, iOther.data_);
      }
      
   private:
      //ESHandle(const ESHandle&); // stop default

      //const ESHandle& operator=(const ESHandle&); // stop default

      // ---------- member data --------------------------------
         const T* data_; 
};

   }
}
#endif /* Framework_ESHandle_h */
