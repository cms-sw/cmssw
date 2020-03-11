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
//

// system include files
#include <map>
#include <string>
#include <typeinfo>
#include <vector>

// user include files
#include "DataFormats/FWLite/interface/IOVSyncValue.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

// forward declarations
class TTree;
class TBranch;
namespace edm {
  class EventID;
  class Timestamp;
}  // namespace edm

namespace cms {
  class Exception;
}

namespace fwlite {

  class Record {
  public:
    Record(const char* iName, TTree*);
    virtual ~Record();

    // ---------- const member functions ---------------------
    const std::string& name() const;

    template <typename HANDLE>
    bool get(HANDLE&, const char* iLabel = "") const;

    const IOVSyncValue& startSyncValue() const;
    const IOVSyncValue& endSyncValue() const;

    std::vector<std::pair<std::string, std::string>> typeAndLabelOfAvailableData() const;
    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void syncTo(const edm::EventID&, const edm::Timestamp&);

  private:
    Record(const Record&) = delete;  // stop default

    const Record& operator=(const Record&) = delete;  // stop default

    cms::Exception* get(const edm::TypeID&, const char* iLabel, const void*&) const;
    void resetCaches();
    // ---------- member data --------------------------------
    std::string m_name;
    TTree* m_tree;
    std::map<IOVSyncValue, unsigned int> m_startIOVtoEntry;
    long m_entry;
    IOVSyncValue m_start;
    IOVSyncValue m_end;

    //This class is not inteded to be used across different threads
    CMS_SA_ALLOW mutable std::map<std::pair<edm::TypeID, std::string>, std::pair<TBranch*, void*>> m_branches;
  };

  template <typename HANDLE>
  bool Record::get(HANDLE& iHandle, const char* iLabel) const {
    const void* value = nullptr;
    cms::Exception* e = get(edm::TypeID(iHandle.typeInfo()), iLabel, value);
    if (nullptr == e) {
      iHandle = HANDLE(value);
    } else {
      iHandle = HANDLE(e);
    }
    return nullptr == e;
  }

}  // namespace fwlite

#endif
