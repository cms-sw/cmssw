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
#include "FWCore/Framework/interface/HCTypeTagTemplate.h"
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class EventSetupRecordKey;
//NOTE: when EDM gets own exception hierarchy, will need to change inheritance
template <class T>
class NoRecordException : public cms::Exception
{
 public:
  // ---------- Constructors and destructor ----------------
  NoRecordException():cms::Exception("NoRecord")
  {
    (*this)
      << "No \"" 
      << heterocontainer::HCTypeTagTemplate<T,EventSetupRecordKey>::className()
      << "\" record found in the EventSetup.\n Please add an ESSource or ESProducer that delivers such a record.\n";
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
