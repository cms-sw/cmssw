#ifndef Fireworks_Core_FWEvePtr_h
#define Fireworks_Core_FWEvePtr_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEvePtr
//
/**\class FWEvePtr FWEvePtr.h Fireworks/Core/interface/FWEvePtr.h

   Description: Smart pointer which properly deals with TEveElement reference counting

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Wed Nov 12 11:08:49 EST 2008
// $Id: FWEvePtr.h,v 1.2 2009/01/23 21:35:41 amraktad Exp $
//

// system include files
#include <boost/shared_ptr.hpp>
#include "TEveElement.h"

// user include files

// forward declarations

template < typename T>
class FWEvePtr {

public:
   FWEvePtr() {
   }
   explicit FWEvePtr(T* iElement) : m_container(new TEveElementList()) {
      m_container->AddElement(iElement);
   }
   // ---------- const member functions ---------------------
   T* operator->() const {
      return m_container && m_container->HasChildren() ?
             static_cast<T*>(m_container->FirstChild()) :
             static_cast<T*>(0);
   }
   T& operator*() const {
      return *(operator->());
   }

   T* get() const {
      return (operator->());
   }

   operator bool() const {
      return m_container && m_container->HasChildren();
   }
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void reset() {
      m_container.reset();
   }
   void reset(T* iNew) {
      FWEvePtr<T> temp(iNew);
      swap(temp);
   }
   void destroyElement() {
      if(m_container) {m_container->DestroyElements();}
      reset();
   }

   void swap(FWEvePtr<T>& iOther) {
      m_container.swap(iOther.m_container);
   }
private:
   //FWEvePtr(const FWEvePtr&); // stop default

   //const FWEvePtr& operator=(const FWEvePtr&); // stop default

   // ---------- member data --------------------------------
   boost::shared_ptr<TEveElement> m_container;
};


#endif
