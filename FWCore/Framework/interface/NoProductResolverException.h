#ifndef FWCore_Framework_NoProductResolverException_h
#define FWCore_Framework_NoProductResolverException_h
// -*- C++ -*-
//
// Package:     Framework
// Module:      NoProductResolverException
//
/**\class NoProductResolverException NoProductResolverException.h FWCore/Framework/interface/NoProductResolverException.h

 Description: An exception that is thrown whenever resolver was not available
              in the EventSetup, it is subset of NoDataException, see more details
              in that class

*/
//
// Author:      Valentine Kouznetsov
// Created:     Wed Apr 23 10:58:26 EDT 2003
//
//

// system include files

// user include files
#include "FWCore/Framework/interface/NoDataException.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

// forward declarations
namespace edm {
  namespace eventsetup {
    template <class T>
    class NoProductResolverException : public NoDataException<T> {
      // ---------- friend classes and functions ---------------

    public:
      // ---------- constants, enums and typedefs --------------

      // ---------- Constructors and destructor ----------------
      NoProductResolverException(const EventSetupRecordKey& iKey, const DataKey& iDataKey)
          : NoDataException<T>(iKey, iDataKey, "NoProductResolverException", NoDataExceptionBase::noProviderMessage()) {}

      // ---------- member functions ---------------------------

    private:
      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- Constructors and destructor ----------------
      //NoProductResolverException(const NoProductResolverException&) ; //allow default

      //const NoProductResolverException& operator=(const NoProductResolverException&); // allow default

      // ---------- data members -------------------------------
    };
  }  // namespace eventsetup
}  // namespace edm
// inline function definitions

#endif
