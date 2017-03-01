#ifndef FWCore_ServiceRegistry_TerminationOrigin_h
#define FWCore_ServiceRegistry_TerminationOrigin_h
// -*- C++ -*-
//
// Package:     FWCore/ServiceRegistry
// Class  :     edm::TerminationOrigin
// 
/**\class edm::TerminationOrigin TerminationOrigin.h "TerminationOrigin.h"

 Description: Enum for different possible origins of a job termination 'signal'

 Usage:
    These values are used to denote exactly why a job is terminating prematurely.

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 02 Sep 2014 20:22:02 GMT
//

// system include files

// user include files

// forward declarations

namespace edm {
  enum class TerminationOrigin {
    ExceptionFromThisContext,
    ExceptionFromAnotherContext,
    ExternalSignal
  };
}

#endif
