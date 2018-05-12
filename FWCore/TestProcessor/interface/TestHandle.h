#ifndef Subsystem_Package_TestHandle_h
#define Subsystem_Package_TestHandle_h
// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     TestHandle
// 
/**\class TestHandle TestHandle.h "TestHandle.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  root
//         Created:  Thu, 03 May 2018 15:50:05 GMT
//

// system include files
#include <memory>

// user include files
#include "DataFormats/Common/interface/BasicHandle.h"
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations

namespace edm {
namespace test {
  template<typename T>
  class TestHandle {
  public:
    explicit TestHandle(T const* product):
    product_{product}
    {}
    
    explicit TestHandle(std::shared_ptr<HandleExceptionFactory> iFailed):
    product_{nullptr},
    whyFailedFactory_{std::move(iFailed)}
    {}
    
    operator bool() {
      return product_!=nullptr;
    }
    
    bool isValid() const {
      return product_ != nullptr;
    }
    
    T const*
    product() const {
      return productStorage();
    }
    
    T const*
    operator->() const {
      return product();
    }
    
    T const&
    operator*() const {
      return *product();
    }
    
  private:
    T const* productStorage() const {
      if(product_) {
        return product_;
      }
      throw *(whyFailedFactory_->make());
    }
    
    T const* product_;
    std::shared_ptr<HandleExceptionFactory> whyFailedFactory_;
  };

}
}

#endif
