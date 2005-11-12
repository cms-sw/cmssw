#ifndef Framework_NoProxyException_h
#define Framework_NoProxyException_h
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
// $Id: NoProxyException.h,v 1.7 2005/09/01 23:30:49 wmtan Exp $
//
//

// system include files
#include <string>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/NoDataException.h"
//#include "DataHandler/interface/FrameRecordItr.h"
//#include "DataHandler/interface/RecordKeyItr.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Utilities/interface/Exception.h"

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
