#if !defined(EVENTSETUP_NOPROXYEXCEPTION_H)
#define EVENTSETUP_NOPROXYEXCEPTION_H
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
// $Id: NoProxyException.h,v 1.3 2005/06/23 22:01:31 wmtan Exp $
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
	 NoDataException<T>(iRecord.key(), iDataKey), 
	 record_(iRecord),
	 message_() {}
      virtual ~NoProxyException() throw() {}

      // ---------- member functions ---------------------------

      // ---------- const member functions ---------------------
      virtual const char* what()const throw() { 
         /*
         std::stringstream m_stream1, m_stream2;
        // Evaluate more precisely what is going on with thrown exception

        // look for proxy in other records
        const Frame& iFrame = record_.frame();
        Frame::const_iterator fIter = iFrame.begin();
        Frame::const_iterator fIEnd = iFrame.end();
        std::string o_record_proxy = "";
        while(fIter != fIEnd) 
        { // loop over all records in current frame
          if(fIter->find(this->dataKey()))
          { // search if proxy exist in other record
            o_record_proxy = "However this data has been found in ";
             m_stream1 << fIter->stream() << " record." << "\0" << std::flush;
            o_record_proxy+= m_stream1.str();
          }
          fIter++;
        }
        
        // search if proxy has another tags
        Record::const_key_iterator pIter = record_.begin_key();
        Record::const_key_iterator iEnd  = record_.end_key();
        std::string sametype_proxy = "";
        while(pIter != iEnd)
        {
          if(pIter->type()  > this->dataKey().type()) 
          {
            break;
          }
          if(pIter->type() == this->dataKey().type()) 
          {
            if(!sametype_proxy.size())
            {
              m_stream2 <<"This data type \"" << pIter->type().name()
                        <<"\" exists, but has different tags.\n";
            }
            m_stream2 <<" usage \"" << pIter->usage().value() << "\""
                      <<" production \""
             << pIter->production().value() << "\""<< "\0" << std::flush;
            sametype_proxy+= m_stream2.str();
          }
          pIter++;
        }
        */
        if(message_.size() == 0) {
          message_ = this->dataTypeMessage();
          /*
          if(o_record_proxy.size()) {
             message_ += std::string(" \n ")+o_record_proxy;
             message_ += std::string(" \n Perhaps you need to change your extract call to use a different record.");
          } else if(sametype_proxy.size()) {
             message_ += std::string(" \n ")+sametype_proxy;
             message_ += std::string(" \n Please check your code and/or scripts for correct usage/production tag.");
          } else {
             */
             message_ += std::string(" \n ")
             +std::string("Please add a Source or Producer to your job which can deliver this data.");
             /*
          }
              */
        }
        return message_.c_str();
      }

      // ---------- static member functions --------------------

   private:
      // ---------- Constructors and destructor ----------------
      //NoProxyException(const NoProxyException&) ; //allow default

      //const NoProxyException& operator=(const NoProxyException&); // allow default

      // ---------- data members -------------------------------
      const EventSetupRecord& record_;
      mutable std::string message_;
      
};
   }
}
// inline function definitions

#endif /* EVENTSETUP_NOPROXYEXCEPTION_H */
