#ifndef FWCore_Framework_ESTransientHandle_h
#define FWCore_Framework_ESTransientHandle_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESTransientHandle
// 
/**\class ESTransientHandle ESTransientHandle.h FWCore/Framework/interface/ESTransientHandle.h

 Description: Provides transient access to data in an EventSetup Record

 Usage:
    This handle is used to setup a memory optimization.  Data obtained via this handle are allowed to be discarded before
the end of the actual IOV for the data.  In this way the system can claim back some memory
 
    Only use this form of the EventSetup handle IF AND ONLY IF
 1) you do not plan on holding onto a pointer to the EventSetup data to which the handle refers
    (since the pointer may become invalid after returning from your function)
 2) you only access this EventSetup data once per IOV change of the Record [i.e. you do NOT read it every Event]
    (failure to do this will cause the EventSetup data to be created each access which can drastically slow the system)
 
 If you are unsure whether to use this handle or not then it is best not to and just use the regular ESHandle.
 
*/
//
// Author:      Chris Jones
// Created:     Thu Nov 12 14:47:35 CST 2009
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESHandle.h"

// forward declarations
namespace edm {

template<typename T>
class ESTransientHandle : public ESHandleBase {
   public:
      typedef T value_type;
   
      ESTransientHandle() : ESHandleBase() {}
      ESTransientHandle(T const* iData) : ESHandleBase(iData, 0) {}
      ESTransientHandle(T const* iData, edm::eventsetup::ComponentDescription const* desc) : ESHandleBase(iData, desc) {}

      // ---------- const member functions ---------------------
      T const* product() const { return static_cast<T const *>(productStorage()); }
      T const* operator->() const { return product(); }
      T const& operator*() const { return *product(); }
      // ---------- static member functions --------------------
      static const bool transientAccessOnly = true;

      // ---------- member functions ---------------------------
      
   private:
};

}
#endif
