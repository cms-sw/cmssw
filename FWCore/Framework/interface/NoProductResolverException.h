// -*- C++ -*-
#ifndef FWCore_Framework_NoProductResolverException_h
#define FWCore_Framework_NoProductResolverException_h
//
// Package:     Framework
// Module:      NoProductResolverException
//
/**\class edm::eventsetup::NoProductResolverException

 Description: An exception that is thrown whenever resolver was not available
              in the EventSetup
*/
//
// Author:      Valentine Kouznetsov
// Created:     Wed Apr 23 10:58:26 EDT 2003
//

#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  namespace eventsetup {

    class NoProductResolverException : public cms::Exception {
    public:
      NoProductResolverException(const EventSetupRecordKey&,
                                 const char* iClassName,
                                 const char* productLabel,
                                 bool moduleLabelDoesNotMatch);
    };

  }  // namespace eventsetup
}  // namespace edm

#endif
