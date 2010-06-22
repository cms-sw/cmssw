#ifndef DataFormats_FWLite_Record_h
#define DataFormats_FWLite_Record_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     Record
// 
/**\class Record Record.h DataFormats/FWLite/interface/Record.h

 Description: Contains conditions data which are related by life-time (i.e. IOV)

 Usage:
    Records are obtained from the fwlite::EventSetup.  
    See DataFormats/FWLite/interface/EventSetup.h for full usage.

*/
//
// Original Author:  
//         Created:  Thu Dec 10 15:58:15 CST 2009
// $Id: Record.h,v 1.1 2009/12/16 17:42:31 chrjones Exp $
//

// system include files
#include <string>
#include <map>

// user include files
#include "DataFormats/FWLite/interface/IOVSyncValue.h"
#include "FWCore/Utilities/interface/TypeIDBase.h"

// forward declarations
class TTree;
class TBranch;
namespace edm {
   class EventID;
   class Timestamp;
}

namespace cms {
   class Exception;
}

namespace fwlite
{
   
   class Record {
      class TypeID : public edm::TypeIDBase {
      public:
         TypeID(const type_info& iInfo): edm::TypeIDBase(iInfo) {}
         using TypeIDBase::typeInfo;
      };

   public:
      Record(const char* iName, TTree*);
      virtual ~Record();

      // ---------- const member functions ---------------------
      const std::string& name() const;
      
      template< typename HANDLE>
      bool get(HANDLE&,const char* iLabel="")const;
      
      const IOVSyncValue& startSyncValue() const;
      const IOVSyncValue& endSyncValue() const;
   
      std::vector<std::pair<std::string,std::string> > typeAndLabelOfAvailableData() const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void syncTo(const edm::EventID&, const edm::Timestamp&);

   private:
      Record(const Record&); // stop default

      const Record& operator=(const Record&); // stop default

      cms::Exception* get(const TypeID&, const char* iLabel, const void*&) const;
      // ---------- member data --------------------------------
      std::string m_name;
      TTree* m_tree;
      std::map<IOVSyncValue,unsigned int> m_startIOVtoEntry;
      long m_entry;
      IOVSyncValue m_start;
      IOVSyncValue m_end;
      
      mutable std::map<std::pair<TypeID,std::string>, TBranch*> m_branches;
   };

   template <typename HANDLE>
   bool
   Record::get(HANDLE& iHandle, const char* iLabel) const
   {
      const void* value = 0;
      cms::Exception* e = get(TypeID(iHandle.typeInfo()),iLabel,value);
      if(0==e){
         iHandle = HANDLE(value);
      } else {
         iHandle = HANDLE(e);
      }
      return 0==e;
   }

} /* fwlite */


#endif
