#ifndef Fireworks_Core_FWHandle_h
#define Fireworks_Core_FWHandle_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWHandle
//
/**\class FWHandle FWHandle.h Fireworks/Core/interface/FWHandle.h

   Description: Used to get a particular data item from a FWEventItem

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Jan  3 19:23:45 EST 2008
// $Id: FWHandle.h,v 1.3 2009/01/23 21:35:41 amraktad Exp $
//

// system include files
#if !defined(__CINT__) && !defined(__MAKECINT__)
//CINT can't handle parsing these files
#include "DataFormats/Common/interface/Wrapper.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#endif

// user include files

// forward declarations

class FWHandle
{

public:
   FWHandle() : data(0) {
   }
   //virtual ~FWHandle();

   // ---------- const member functions ---------------------
   const T* get() const {
      return data_;
   }

   const T* operator->() const {
      return data_;
   }

   const T& operator*() const {
      return *data_;
   }
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void getFrom(const FWEventItem& iItem) {
      data_ = reinterpet_cast<const T*>(
         iItem.data(edm::Wrapper<T>::productTypeInfo())
         );
   }

private:
   //FWHandle(const FWHandle&); // stop default

   //const FWHandle& operator=(const FWHandle&); // stop default

   // ---------- member data --------------------------------
   const T* data_;

};


#endif
