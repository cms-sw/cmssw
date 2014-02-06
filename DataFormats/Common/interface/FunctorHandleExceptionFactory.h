#ifndef Subsystem_Package_FunctorHandleExceptionFactory_h
#define Subsystem_Package_FunctorHandleExceptionFactory_h
// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     FunctorHandleExceptionFactory
// 
/**\class FunctorHandleExceptionFactory FunctorHandleExceptionFactory.h "FunctorHandleExceptionFactory.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 04 Dec 2013 18:41:59 GMT
//

// system include files
#include "DataFormats/Common/interface/HandleExceptionFactory.h"

// user include files

// forward declarations
namespace edm {
  
  template<typename T>
  class FunctorHandleExceptionFactory : public HandleExceptionFactory
  {
    
  public:
    FunctorHandleExceptionFactory(T&& iFunctor) : m_functor(std::move(iFunctor)) {}

    //FunctorHandleExceptionFactory(T iFunctor) : m_functor(iFunctor) {}

    // ---------- const member functions ---------------------

    std::shared_ptr<cms::Exception> make() const {
      return m_functor();
    }
  private:
    T m_functor;
  };

  template<typename T>
  std::shared_ptr<HandleExceptionFactory> makeHandleExceptionFactory(T&& iFunctor) {
    return std::make_shared<FunctorHandleExceptionFactory<T>>(std::move(iFunctor));
  }
}


#endif
