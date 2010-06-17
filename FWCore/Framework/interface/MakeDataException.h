#ifndef FWCore_Framework_MakeDataException_h
#define FWCore_Framework_MakeDataException_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     MakeDataException
// 
/**\class MakeDataException MakeDataException.h FWCore/Framework/interface/MakeDataException.h

Description: An exception that is thrown whenever a Proxy had a problem with
its algorithm.

 Usage:
This exception will be thrown automatically if a a class that inherits from
DataProxyTemplate<> returns 0 from its make method.

If you wish to explain the reason for the error, you can throw a 
MakeDataException from within your Proxy
E.g.
\code
if(outOfBoundsValue) {
   throw MakeDataException(" value out of bounds",
                           MakeDataExceptionInfo<record_type, value_type>(iDataKey));
}
\endcode

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  1 13:18:53 EST 2005
//

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/HCTypeTag.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations
namespace edm {
   namespace eventsetup {

class MakeDataException : public cms::Exception
{
   public:
      MakeDataException(const EventSetupRecordKey&, const DataKey&);  
      ~MakeDataException() throw() {}

      // ---------- const member functions ---------------------
      const char* myMessage() const throw() {
         return message_.c_str();
      }
   
      // ---------- static member functions --------------------
      static std::string standardMessage(const EventSetupRecordKey&, const DataKey&); 
   // ---------- member functions ---------------------------

   private:
      //MakeDataException(const MakeDataException&); // stop default

      //const MakeDataException& operator=(const MakeDataException&); // stop default

      // ---------- member data --------------------------------
      std::string message_;
};

   }
}
#endif
