#ifndef EVENTSETUP_MAKEDATAEXCEPTION_H
#define EVENTSETUP_MAKEDATAEXCEPTION_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     MakeDataException
// 
/**\class MakeDataException MakeDataException.h Core/CoreFramework/interface/MakeDataException.h

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
   throw MakeDataException<record_type, value_type>(
                                                    " value out of bounds",
                                                    iDataKey);
}
\endcode

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  1 13:18:53 EST 2005
// $Id: MakeDataException.h,v 1.1 2005/05/29 02:29:53 wmtan Exp $
//

// system include files
#include <string>
#include <exception>

// user include files
#include "FWCore/CoreFramework/interface/HCTypeTagTemplate.h"
#include "FWCore/CoreFramework/interface/DataKey.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordKey.h"

// forward declarations
namespace edm {
   namespace eventsetup {
template< class RecordT, class DataT>
class MakeDataException : public std::exception
{

   public:
      MakeDataException(const DataKey& iKey) : 
        message_(standardMessage(iKey)){}

      MakeDataException(const std::string& iAdditionalInfo,
                     const DataKey& iKey) : 
        message_(messageWithInfo(iKey, iAdditionalInfo)){}

      ~MakeDataException() throw() {}

      // ---------- const member functions ---------------------
      const char* what() const throw() {
         return message_.c_str();
      }
   
      // ---------- static member functions --------------------
      static std::string standardMessage(const DataKey& iKey) {
         std::string returnValue = std::string("Error while making data ") 
         +"\""
         +heterocontainer::HCTypeTagTemplate<DataT,DataKey>::className() 
         +"\" "
         +"\""
         +iKey.name().value()
         +"\" "
         +"in Record "
         +heterocontainer::HCTypeTagTemplate<RecordT, EventSetupRecordKey>::className();
         return returnValue;
      }
   
      static std::string messageWithInfo(const DataKey& iKey,
                                          const std::string& iInfo) {
         return standardMessage(iKey) +"\n"+iInfo;
      }
   // ---------- member functions ---------------------------

   private:
      //MakeDataException(const MakeDataException&); // stop default

      //const MakeDataException& operator=(const MakeDataException&); // stop default

      // ---------- member data --------------------------------
      std::string message_;
      
};

   }
}
#endif /* EVENTSETUP_MAKEDATAEXCEPTION_H */
