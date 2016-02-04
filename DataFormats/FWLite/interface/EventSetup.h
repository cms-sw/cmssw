#ifndef DataFormats_FWLite_EventSetup_h
#define DataFormats_FWLite_EventSetup_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     EventSetup
// 
/**\class EventSetup EventSetup.h DataFormats/FWLite/interface/EventSetup.h

 Description: Provides access to conditions information from fwlite

 Usage:
    This class provides a friendly interface for accessing conditions information
    which have been stored into a ROOT TFile.
    
    As in the full framework, conditions data are collected in the EventSetup.  The
    EventSetup holds 'Records' where each 'Record' holds data where all the data
    in one 'Record' is valid for the same period of time (referred to as an 
    'Interval of Validity' or IOV for short).
    
    The normal usage of this class is as follows
    
    TFile condFile("conditions.root");
    
    fwlite::EventSetup es(&condFile);
    
    fwlite::RecordID fooID = es.recordID("FooRecord");
    
    for(...) {  //looping over some event data
       //eventID and timestamp are obtained from the event
       es.syncTo(eventID, timestamp);
       
       fwlite::ESHandle<Foo> fooHandle;
       es.get(fooID).get(fooHandle);
       
       //now access the info in Foo
      std::cout << fooHandle->value()<<std::endl;
    }

*/
//
// Original Author:  
//         Created:  Thu Dec 10 15:57:46 CST 2009
//

// system include files
#include <vector>
#include <string>

// user include files
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

// forward declarations
class TFile;

namespace edm {
   class EventBase;
}

namespace fwlite
{
   class Record;
   typedef unsigned int RecordID;
   
   class EventSetup {
      
   public:
      EventSetup(TFile*);
      virtual ~EventSetup();
      
      // ---------- const member functions ---------------------
      const Record& get(const RecordID&) const;

      /**Returns the lookup id of the record whose name is iRecordName.  The returned id
      is only valid for the instance of an EventSetup object to which the recordID call was made.
      If you later create a new EventSetup instance even for the same file the RecordIDs can be different.
      */
      RecordID recordID(const char* iRecordName) const;

      /**Returns true if a record with the name iRecordName is available in the file
      */
      bool exists(const char* iRecordName) const;
      
      std::vector<std::string> namesOfAvailableRecords() const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      //void syncTo(unsigned long iRun, unsigned long iLumi);
      //void syncTo(const edm::EventID&);

      /** Ensures that all Records will access the appropriate data for this instant in time
      */
      void syncTo(const edm::EventID&, const edm::Timestamp&);

      //void autoSyncTo(const edm::EventBase&);
   

   private:
      EventSetup(const EventSetup&); // stop default

      const EventSetup& operator=(const EventSetup&); // stop default

      // ---------- member data --------------------------------
      const edm::EventBase* m_event;
      edm::EventID m_syncedEvent;
      edm::Timestamp m_syncedTimestamp;
      
      TFile* m_file;

      mutable std::vector<Record*> m_records;
   };
} /* fwlite */


#endif
