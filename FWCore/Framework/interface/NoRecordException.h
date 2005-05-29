#if !defined(EVENTSETUP_NORECORDEXCEPTION_H)
#define EVENTSETUP_NORECORDEXCEPTION_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Module:      NoRecordException
// 
/**\class NoRecordException NoRecordException.h CoreFramework/interface/NoRecordException.h

 Description: An exception that is thrown whenever a EventSetup is asked to retrieve
              a Record it does not have.

 Usage:
   This exception will be thrown if you call the EventSetup method get() and the
    record type you request does not exist.
    E.g.
    \code
    try {
      iEventSetup.get<MyRecord>()...;
    } catch( eventsetup::NoRecordException& iException ) {
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
#include <exception>
// user include files
#include "FWCore/CoreFramework/interface/HCTypeTagTemplate.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class EventSetupRecordKey;
//NOTE: when EDM gets own exception hierarchy, will need to change inheritance
template <class T>
class NoRecordException : public std::exception
{
   public:
      // ---------- Constructors and destructor ----------------
      NoRecordException() {
	    message_ = std::string("No ") +
         heterocontainer::HCTypeTagTemplate<T,EventSetupRecordKey>::className() +
	       " Record found in the EventSetup";
      }
      virtual ~NoRecordException() throw() {}

      // ---------- const member functions ---------------------
      const char* what() const throw() {
	 return message_.c_str();
      }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   
   private:
      // ---------- Constructors and destructor ----------------
      //NoRecordException( const NoRecordException& ); // stop default

      // ---------- assignment operator(s) ---------------------
      //const NoRecordException& operator=( const NoRecordException& ); // stop default

      // ---------- data members -------------------------------
      std::string message_;

};

// inline function definitions
   }
}
#endif /* EVENTSETUP_NORECORDEXCEPTION_H */
