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
#include "FWCore/Framework/interface/HCTypeTagTemplate.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations
namespace edm {
   namespace eventsetup {

class MakeDataExceptionInfoBase
{
public:
  MakeDataExceptionInfoBase(const DataKey &iKey, const std::string& dataClassName, const std::string &recordClassName)
  :key_(iKey),
   dataClassName_(dataClassName),
   recordClassName_(recordClassName)
  {}
  const DataKey &key() const {return key_;}
  const std::string &dataClassName() const {return dataClassName_;}
  const std::string &recordClassName() const {return recordClassName_;}
private:
  DataKey key_;
  std::string dataClassName_;
  std::string recordClassName_;
};

template <class RecordT, class DataT>
class MakeDataExceptionInfo : public MakeDataExceptionInfoBase
{
public:
  MakeDataExceptionInfo(const DataKey &iKey) 
  : MakeDataExceptionInfoBase(iKey, 
                              heterocontainer::HCTypeTagTemplate<DataT,DataKey>::className(), 
                              heterocontainer::HCTypeTagTemplate<RecordT, EventSetupRecordKey>::className()) {}
};

class MakeDataException : public cms::Exception
{
   public:
      MakeDataException(const MakeDataExceptionInfoBase &info);  
      MakeDataException(const std::string& iAdditionalInfo,
                        const MakeDataExceptionInfoBase& info);  
      ~MakeDataException() throw() {}

      // ---------- const member functions ---------------------
      const char* myMessage() const throw() {
         return message_.c_str();
      }
   
      // ---------- static member functions --------------------
      static std::string standardMessage(const MakeDataExceptionInfoBase& info); 
      static std::string messageWithInfo(const MakeDataExceptionInfoBase& info,
                                         const std::string& iInfo); 
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
