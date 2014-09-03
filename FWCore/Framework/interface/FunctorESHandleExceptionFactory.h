#ifndef FWCore_Framework_FunctorESHandleExceptionFactory_h
#define FWCore_Framework_FunctorESHandleExceptionFactory_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     FunctorESHandleExceptionFactory
//
/**\class edm::FunctorESHandleExceptionFactory

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  W. David Dagenhart
//         Created:  1 May 2014
//

#include "FWCore/Framework/interface/ESHandleExceptionFactory.h"

#include <exception>
#include <memory>
#include <utility>

namespace edm {

  template<typename T>
  class FunctorESHandleExceptionFactory : public ESHandleExceptionFactory {

  public:
    FunctorESHandleExceptionFactory(T&& iFunctor) : m_functor(std::move(iFunctor)) {}

    std::exception_ptr make() const {
      return m_functor();
    }
  private:
    T m_functor;
  };

  template<typename T>
  std::shared_ptr<ESHandleExceptionFactory> makeESHandleExceptionFactory(T&& iFunctor) {
    return std::make_shared<FunctorESHandleExceptionFactory<T>>(std::move(iFunctor));
  }
}
#endif
