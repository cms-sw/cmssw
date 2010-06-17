#ifndef FWCore_Framework_NoProxyException_h
#define FWCore_Framework_NoProxyException_h
// -*- C++ -*-
//
// Package:     Framework
// Module:      NoProxyException
// 
/**\class NoProxyException NoProxyException.h FWCore/Framework/interface/NoProxyException.h

 Description: An exception that is thrown whenever proxy was not available
              in the EventSetup, it is subset of NoDataException, see more details
              in that class

*/
//
// Author:      Valentine Kouznetsov
// Created:     Wed Apr 23 10:58:26 EDT 2003
// $Id: NoProxyException.h,v 1.10 2009/07/02 16:46:50 chrjones Exp $
//
//

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/NoDataException.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

// forward declarations
namespace edm {
   namespace eventsetup {
template <class T>
class NoProxyException : public NoDataException<T>
{
      // ---------- friend classes and functions ---------------

   public:
      // ---------- constants, enums and typedefs --------------

      // ---------- Constructors and destructor ----------------
      NoProxyException(const EventSetupRecordKey& iKey,
			  const DataKey& iDataKey) :
       NoDataException<T>(iKey, iDataKey,"NoProxyException",NoDataExceptionBase::noProxyMessage()) 
       {
       }

      // ---------- member functions ---------------------------

   private:
      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- Constructors and destructor ----------------
      //NoProxyException(const NoProxyException&) ; //allow default

      //const NoProxyException& operator=(const NoProxyException&); // allow default

      // ---------- data members -------------------------------      
};
   }
}
// inline function definitions

#endif
