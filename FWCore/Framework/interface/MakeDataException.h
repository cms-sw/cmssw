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
   throw MakeDataException<record_type, value_type>(
                                                    " value out of bounds",
                                                    iDataKey);
}
\endcode

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  1 13:18:53 EST 2005
// $Id: MakeDataException.h,v 1.6 2005/09/01 23:30:49 wmtan Exp $
//

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/HCTypeTagTemplate.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations
namespace edm {
   namespace eventsetup {
template< class RecordT, class DataT>
class MakeDataException : public cms::Exception
{

   public:
      MakeDataException(const DataKey& iKey) : 
	cms::Exception("MakeDataException"),
        message_(standardMessage(iKey))
      {
	this->append(myMessage());
      }

      MakeDataException(const std::string& iAdditionalInfo,
                     const DataKey& iKey) : 
        message_(messageWithInfo(iKey, iAdditionalInfo))
      {
	this->append(this->myMessage());
      }

      ~MakeDataException() throw() {}

      // ---------- const member functions ---------------------
      const char* myMessage() const throw() {
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
#endif
