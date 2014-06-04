#ifndef FWCore_Framework_ESHandleExceptionFactory_h
#define FWCore_Framework_ESHandleExceptionFactory_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ESHandleExceptionFactory
//
/**\class edm::ESHandleExceptionFactory

 Description: Creates exceptions for an edm::ESHandle

 Usage:
    When an event setup data product is requested which is not available,
 the edm::ESHandle<> will hold an edm::ESHandleExceptionFactory which is
 used to manufacture the exception which will be thrown if the ESHandle
 is dereferenced.
 Using a factory over having the exception already in the ESHandle
 is faster for the case where code calls Handle::isValid before
 dereferencing.

*/
//
// Original Author:  W. David Dagenhart
//         Created:  1 May 2014
//

#include <exception>

namespace edm {

  class ESHandleExceptionFactory
  {

  public:
    ESHandleExceptionFactory();
    virtual ~ESHandleExceptionFactory();

    virtual std::exception_ptr make() const = 0;
  };
}
#endif
