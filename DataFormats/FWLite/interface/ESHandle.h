#ifndef DataFormats_FWLite_ESHandle_h
#define DataFormats_FWLite_ESHandle_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     ESHandle
// 
/**\class ESHandle ESHandle.h DataFormats/FWLite/interface/ESHandle.h

 Description: Used with fwlite::Record to retrieve conditions information

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Dec 14 15:16:07 CST 2009
//

// system include files
#include <memory>
#include <typeinfo>
#include <vector>

// user include files
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations
namespace fwlite {
   class Record;
   
   std::shared_ptr<cms::Exception> eshandle_not_set_exception();
   
template <class T>
class ESHandle
{
   friend class fwlite::Record;
   
   public:
      ESHandle(): m_data(nullptr), m_exception(eshandle_not_set_exception()) {}

      // ---------- const member functions ---------------------
      bool isValid() const { return nullptr != m_data;}

      const T& operator*() const {
         if(nullptr != m_exception.get()) {
            throw *m_exception;
         }
         return *m_data;
      }
      
      const T* operator->() const {
         if(nullptr != m_exception.get()) {
            throw *m_exception;
         }
         return m_data;
         
      }
      
      const std::type_info& typeInfo() const {
         return typeid(T);
      }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      ESHandle(const void* iData) :
         m_data(static_cast<const T*>(iData)),
         m_exception() {}
      ESHandle(cms::Exception* iException) :
         m_data(nullptr), m_exception(iException) {}
      //ESHandle(const ESHandle&); // stop default

      //const ESHandle& operator=(const ESHandle&); // stop default

      // ---------- member data --------------------------------
      const T* m_data;
      std::shared_ptr<cms::Exception> m_exception;
};

}
#endif
