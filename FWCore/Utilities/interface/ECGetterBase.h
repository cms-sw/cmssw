#ifndef Utilities_ECGetterBase_h
#define Utilities_ECGetterBase_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     ECGetterBase
// 
/**\class ECGetterBase ECGetterBase.h FWCore/Utilities/interface/ECGetterBase.h

 Description: Helper class for the implementation of edm::ExtensionCord

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep 22 12:41:05 EDT 2006
//

// system include files

// user include files

// forward declarations
namespace edm {
  namespace extensioncord {
    template <class T>
    class ECGetterBase
    {
      
     public:
      ECGetterBase(): data_(0) {}
      virtual ~ECGetterBase() {}

      // ---------- const member functions ---------------------
      const T* get() const  {
        if(data_==0) {
          data_ = this->getImpl();
        }
        return data_;
      }
      
     private:
      //ECGetterBase(const ECGetterBase&); // allow default

      //const ECGetterBase& operator=(const ECGetterBase&); // allow default
      
      //the actual method which is used to get the requested data
      virtual const T* getImpl() const = 0;

      // ---------- member data --------------------------------
      //does not own the data
      mutable const T* data_;

    };
  }
  template <class T>
  class ValueHolderECGetter : public extensioncord::ECGetterBase<T> {
  public:
    ValueHolderECGetter(const T& iValue) : value_(&iValue) {}
  private:
    virtual const T* getImpl() const {
      return value_;
    }
    const T* value_;
  };
}

#endif
