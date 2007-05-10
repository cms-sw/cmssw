#ifndef DataFormats_FWLite_Event_h
#define DataFormats_FWLite_Event_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     Event
// 
/**\class Event Event.h DataFormats/FWLite/interface/Event.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue May  8 15:01:20 EDT 2007
// $Id$
//

// system include files
#include <typeinfo>
#include <map>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "Rtypes.h"
#include "Reflex/Object.h"

// user include files
#include "FWCore/Utilities/interface/TypeID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

// forward declarations
namespace fwlite {
  namespace internal {
  class DataKey {
public:
    //NOTE: Do not take ownership of strings.  This is done to avoid
    // doing 'new's and string copies when we just want to lookup the data
    // This means something else is responsible for the pointers remaining
    // valid for the time for which the pointers are still in use
    DataKey(const edm::TypeID& iType,
            const char* iModule,
            const char* iProduct,
            const char* iProcess) :
    type_(iType),
    module_(iModule!=0? iModule:kEmpty),
    product_(iProduct!=0?iProduct:kEmpty),
    process_(iProcess!=0?iProcess:kEmpty) {}
    
    ~DataKey() {
    }

    bool operator<( const DataKey& iRHS) const {
      if( type_ < iRHS.type_) {
        return true;
      }
      if( iRHS.type_ < type_ ) {
        return false;
      }
      int comp = std::strcmp(module_,iRHS.module_);
      if( 0!= comp) {
        return comp <0;
      }
      comp = std::strcmp(product_,iRHS.product_);
      if( 0!= comp) {
        return comp <0;
      }
      comp = std::strcmp(process_,iRHS.process_);
      return comp <0;
    }
    static const char* const kEmpty;
    const  char* module() const {return module_;}
    const char* product() const {return product_;}
    const char* process() const {return process_;}
    const edm::TypeID& typeID() const {return type_;}
    
private:
    edm::TypeID type_;
    const char* module_;
    const char* product_;
    const char* process_;
  };
  
  struct Data {
    TBranch* branch_;
    Long64_t lastEvent_;
    ROOT::Reflex::Object obj_;
  };
  }
class Event
{

   public:
      /**NOTE: Does NOT take ownership so iFile must remain around at least as long as Event
  */
      Event(TFile* iFile);
      virtual ~Event();

      const Event& operator++();

      const Event& to(Long64_t);
      
      /** Go to the very first Event*/
      const Event& toBegin();
      
      // ---------- const member functions ---------------------
      /** This function should only be called by fwlite::Handle<>*/
      void getByLabel(const std::type_info&, const char*, const char*, const char*, void*&) const;
      //void getByBranchName(const std::type_info&, const char*, void*&) const;

      bool isValid() const;
      operator bool () const;
      bool atEnd() const;
      
      Long64_t size() const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      Event(const Event&); // stop default

      const Event& operator=(const Event&); // stop default

      const edm::ProcessHistory& history() const;
      // ---------- member data --------------------------------
      TFile* file_;
      TTree* eventTree_;
      Long64_t eventIndex_;

      mutable std::map<internal::DataKey, internal::Data> data_;
      //takes ownership of the strings used by the DataKey keys in data_
      mutable std::vector<const char*> labels_;
      mutable edm::ProcessHistoryMap historyMap_;
      mutable edm::EventAuxiliary aux_;
      edm::EventAuxiliary* pAux_;
      TBranch* auxBranch_; 
};

}
#endif
