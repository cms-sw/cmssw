#ifndef EVENTSETUP_ESHANDLE_H
#define EVENTSETUP_ESHANDLE_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     ESHandle
// 
/**\class ESHandle ESHandle.h FWCore/CoreFramework/interface/ESHandle.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  1 14:47:35 EST 2005
// $Id: ESHandle.h,v 1.1 2005/04/02 14:18:01 chrjones Exp $
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
      ESHandle(const T* iData ) : data_(iData) {}
      //virtual ~ESHandle();

      // ---------- const member functions ---------------------
      const T* product() const { return data_; }
      const T* operator->() const { return product(); }
      const T& operator*() const { return *product(); }
      
      bool isValid() const { return 0 != data_; }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void swap( ESHandle<T>& iOther) {
         std::swap(data_, iOther.data_ );
      }
      
   private:
      //ESHandle( const ESHandle& ); // stop default

      //const ESHandle& operator=( const ESHandle& ); // stop default

      // ---------- member data --------------------------------
         const T* data_; 
};

   }
}
#endif /* EVENTSETUP_ESHANDLE_H */
