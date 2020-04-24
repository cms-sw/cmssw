#ifndef DataFormats_Common_HandleExceptionFactory_h
#define DataFormats_Common_HandleExceptionFactory_h
// -*- C++ -*-
//
// Package:     DataFormats/Common
// Class  :     HandleExceptionFactory
// 
/**\class edm::HandleExceptionFactory HandleExceptionFactory.h "DataFormats/Common/interface/HandleExceptionFactory.h"

 Description: Creates an cms::Exception for an edm::Handle

 Usage:
    When a data product is requested which is not available, the
 edm::Handle<> will hold an edm::HandleExceptionFactory which is
 used to manufacture the exception which will be thrown if the Handle
 is dereferenced.
 Using a factory over having the cms::Exception already in the Handle
 is faster for the case where code calls Handle::isValid before 
 dereferencing.

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 04 Dec 2013 16:47:12 GMT
//

// system include files
#include <memory>

// user include files

// forward declarations

namespace cms {
  class Exception;
}

namespace edm {
  
  class HandleExceptionFactory
  {
    
  public:
    HandleExceptionFactory();
    virtual ~HandleExceptionFactory();
    
    // ---------- const member functions ---------------------
    virtual std::shared_ptr<cms::Exception> make() const = 0;
    
  private:
    //HandleExceptionFactory(const HandleExceptionFactory&); // stop default
    
    //const HandleExceptionFactory& operator=(const HandleExceptionFactory&); // stop default
    
    // ---------- member data --------------------------------
    
  };
}


#endif
