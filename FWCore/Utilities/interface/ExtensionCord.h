#ifndef Utilities_ExtensionCord_h
#define Utilities_ExtensionCord_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     ExtensionCord
// 
/**\class ExtensionCord ExtensionCord.h FWCore/Utilities/interface/ExtensionCord.h

 Description: Allows passing data from an edm::OutletBase to the holder of the ExtensionCord

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep 22 12:38:58 EDT 2006
//

// system include files
#include <boost/shared_ptr.hpp>

// user include files
#include "FWCore/Utilities/interface/ECGetterBase.h"
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations

namespace edm {
  template <class T> class OutletBase;
  
  template <class T>
  class ExtensionCord
  {
    //only something that inherits from OutletBase<T> will
    // be allowed to call 'setGetter'
    friend class OutletBase<T>;
    
   public:
      struct Holder {
        extensioncord::ECGetterBase<T>* getter_;
      };

    
      ExtensionCord(): holder_(new Holder()) {}
      //virtual ~ExtensionCord();

      // ---------- const member functions ---------------------
      const T* operator->() const {
        return this->get();
      }
      
      const T* get() const {
        if (!this->connected()) {
          throw cms::Exception("InvalidExtensionCord")<<"an edm::ExtensionCord for type "<<typeid(T).name()
          <<" was not connected to an outlet. This is a programming error.";
        }
        return holder_->getter_->get();
      }
        
      const T& operator*() const {
        return *(this->get());
      }

      ///Returns true if the ExtensionCord is connected to an outlet and can therefore deliver data
      bool connected() const {
        return 0 != holder_->getter_;
      }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      //ExtensionCord(const ExtensionCord&); // allow default

      //const ExtensionCord& operator=(const ExtensionCord&); // allow default

        void setGetter(extensioncord::ECGetterBase<T>* iGetter ) {
        holder_->getter_ = iGetter;
      }
      // ---------- member data --------------------------------
      boost::shared_ptr< Holder > holder_;
      
  };
}

#endif
