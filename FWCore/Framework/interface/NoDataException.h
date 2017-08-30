#ifndef FWCore_Framework_NoDataException_h
#define FWCore_Framework_NoDataException_h
// -*- C++ -*-
//
// Package:     Framework
// Module:      NoDataException
// 
/**\class NoDataException NoDataException.h Exception/interface/NoDataException.h

 Description: An exception that is thrown whenever data was not available
              in the Frame

 Usage:
    NoDataException<> is thrown whenever an extract call fails because 
    the type of data being extract was not available in the Frame.

    If your program should continue even if the extract call fails, you should
    catch this exception.

    \code
    try {
      Item<DBEventHeader> eventHeader;
      extract(iFrame.record(Stream::kBeginRun), eventHeader);

      report(INFO, kFacilityString) << "run # " << eventHeader->runNumber()
                                     << "event # " << eventHeader->number()
				     << std::endl;

    } catch(NoDataException<Item<DBEventHeader>::contents> &iException) {
      report(WARNING, kFacilityString) << iException.what() << std::endl;
    }
      
    \endcode

    To make it easier to catch exceptions, all of the FAXXX types provide
    C preprocessor macros of the form
    \code
       NO_XXX_DATA_EXCEPTION(type)
    \endcode
     which are just shorthand ways of writing
     \code
       NoDataException<FAXXX<type>::contents>
     \endcode
    E.g.
       \code
       NO_ITEM_DATA_EXCEPTION(DBEventHeader)
       \endcode
       is the same as writing
       \code
       NoDataException<Item<DBEventHeader>::value_type>
       \endcode

    NOTE: NoDataException<> is only thrown when the data is unavailable. If
      the data should have been available but a problem occurred while obtaining
      the data, then a different type of exception will be thrown.

      To catch ALL possible exceptions that can occur from the Data Access 
      system you should catch exceptions of the type DAExceptionBase.
*/
//
// Author:      Chris D Jones
// Created:     Tue Dec  7 09:10:34 EST 1999
//

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/HCTypeTag.h"
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations
namespace edm {
   namespace eventsetup {

class NoDataExceptionBase : public cms::Exception
{
public:
  NoDataExceptionBase(const EventSetupRecordKey& iRecordKey,
                        const DataKey& iDataKey,
                        const char* category_name = "NoDataException") ;
  ~NoDataExceptionBase() noexcept override;
  const DataKey& dataKey() const;
protected:
  static std::string providerButNoDataMessage(const EventSetupRecordKey& iKey);
  static std::string noProxyMessage();
  void constructMessage(const char* iClassName, const std::string& iExtraInfo);
private:
  void beginDataTypeMessage(std::string&) const;
  void endDataTypeMessage(std::string&) const;
  
  // ---------- Constructors and destructor ----------------
  //NoDataExceptionBase(const NoDataExceptionBase&) ; //allow default
  //const NoDataExceptionBase& operator=(const NoDataExceptionBase&); // allow default

  // ---------- data members -------------------------------
  EventSetupRecordKey record_;
  DataKey dataKey_;
};

template <class T>
 class NoDataException : public NoDataExceptionBase 
{
public:
  NoDataException(const EventSetupRecordKey& iRecordKey,
                  const DataKey& iDataKey,
                  const char* category_name = "NoDataException") :
  NoDataExceptionBase(iRecordKey, iDataKey, category_name)
  {
    constructMessage(heterocontainer::className<T>(),
                     providerButNoDataMessage(iRecordKey));
  }

  NoDataException(const EventSetupRecordKey& iRecordKey,
                  const DataKey& iDataKey,
                  const char* category_name ,
                  const std::string& iExtraInfo ) :
  NoDataExceptionBase(iRecordKey, iDataKey, category_name)  
  {
    constructMessage(heterocontainer::className<T>(),
                     iExtraInfo);
  }

};

   }
}
#endif
