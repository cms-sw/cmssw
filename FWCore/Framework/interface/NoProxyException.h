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
// $Id: NoProxyException.h,v 1.8 2005/11/12 16:18:04 chrjones Exp $
//
//

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/NoDataException.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"

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
      NoProxyException(const EventSetupRecord& iRecord,
			  const DataKey& iDataKey) :
	NoDataException<T>(iRecord.key(), iDataKey,"NoProxyException",standardMessage()) 
       {
       }
      virtual ~NoProxyException() throw() {}

      // ---------- member functions ---------------------------

   private:
      // ---------- const member functions ---------------------
      std::string standardMessage()const throw() { 
         return std::string("Please add an ESSource or ESProducer to your job which can deliver this data.\n");
      }

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
