#ifndef FWCore_Framework_NoRecordException_h
#define FWCore_Framework_NoRecordException_h
// -*- C++ -*-
//
// Package:     Framework
// Module:      NoRecordException
// 
/**\class NoRecordException NoRecordException.h Framework/interface/NoRecordException.h

 Description: An exception that is thrown whenever a EventSetup is asked to retrieve
              a Record it does not have.

 Usage:
   This exception will be thrown if you call the EventSetup method get() and the
    record type you request does not exist.
    E.g.
    \code
    try {
      iEventSetup.get<MyRecord>()...;
    } catch(eventsetup::NoRecordException& iException) {
       //no record of type MyRecord found in EventSetup
       ...
    }
    \endcode  
*/
//
// Author:      Chris D Jones
// Created:     Sat Mar 26 10:31:01 EST 2005
//

// system include files
#include <string>
// user include files
#include "FWCore/Framework/interface/HCTypeTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class EventSetupRecordKey;
      void no_record_exception_message_builder(cms::Exception&,const char*);
      void no_dependent_record_exception_message_builder(cms::Exception&, const EventSetupRecordKey&, const char*);
//NOTE: when EDM gets own exception hierarchy, will need to change inheritance
template <class T>
class NoRecordException : public cms::Exception
{
 public:
  // ---------- Constructors and destructor ----------------
  NoRecordException():cms::Exception("NoRecord")
  {
    no_record_exception_message_builder(*this,heterocontainer::className<T>());
  }


  NoRecordException(const EventSetupRecordKey& iKey):cms::Exception("NoRecordFromDependentRecord")
  {
    no_dependent_record_exception_message_builder(*this,iKey,heterocontainer::className<T>());
  }
      virtual ~NoRecordException() throw() {}

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   
   private:
      // ---------- Constructors and destructor ----------------
      //NoRecordException(const NoRecordException&); // stop default

      // ---------- assignment operator(s) ---------------------
      //const NoRecordException& operator=(const NoRecordException&); // stop default

      // ---------- data members -------------------------------
};

// inline function definitions
   }
}
#endif
