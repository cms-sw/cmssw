#if !defined(EVENTSETUP_NODATAEXCEPTION_H)
#define EVENTSETUP_NODATAEXCEPTION_H
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
				     << endl;

    } catch(NoDataException<Item<DBEventHeader>::contents> &iException) {
      report(WARNING, kFacilityString) << iException.what() << endl;
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
      the data should have been available but a problem occured while obtaining
      the data, then a different type of exception will be thrown.

      To catch ALL possible exceptions that can occur from the Data Access 
      system you should catch exceptions of the type DAExceptionBase.
*/
//
// Author:      Chris D Jones
// Created:     Tue Dec  7 09:10:34 EST 1999
// $Id: NoDataException.h,v 1.3 2005/06/23 22:01:31 wmtan Exp $
//

// system include files
#include <string>
#include <exception>

// user include files
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/HCTypeTagTemplate.h"

// forward declarations
namespace edm {
   namespace eventsetup {

template <class T>
class NoDataException : public std::exception
{
      // ---------- friend classes and functions ---------------

   public:
      // ---------- constants, enums and typedefs --------------

      // ---------- Constructors and destructor ----------------
      NoDataException(const EventSetupRecordKey& iRecordKey,
			 const DataKey& iDataKey) : 
	 record_(iRecordKey),
	 dataKey_(iDataKey),
	 message_() {}
      virtual ~NoDataException() throw() {}

      // ---------- const member functions ---------------------
      const DataKey& dataKey() const { return dataKey_; }
      virtual const char* what() const throw() { 
        if(message_.size() == 0) {
          message_ = dataTypeMessage();
           message_+= std::string(" \n ")
                 +" A provider for this data exists, but it's unable to deliver the data for this "
                 +record_.name()
                 +" record.";
        }
	 return message_.c_str();
      }


      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   
   protected:
      const std::string& dataTypeMessage () const { 
	 if(dataTypeMessage_.size() == 0) {

	    dataTypeMessage_ = std::string("No data of type ") 
	       +"\""
            +heterocontainer::HCTypeTagTemplate<T,DataKey>::className() 
	       +"\" "
	       +"\""
	         +dataKey_.name().value() 
	       +"\" "
	       +"in Record "
	       +record_.name();
	 }
	 return dataTypeMessage_;
      }

   private:
      // ---------- Constructors and destructor ----------------
      //NoDataException(const NoDataException&) ; //allow default

      //const NoDataException& operator=(const NoDataException&); // allow default

      // ---------- data members -------------------------------
      EventSetupRecordKey record_;
      DataKey dataKey_;
      mutable std::string message_;
      mutable std::string dataTypeMessage_;
      
};

   }
}
#endif /* EVENTSETUP_NODATAEXCEPTION_H */
