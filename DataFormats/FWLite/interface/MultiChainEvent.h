#ifndef DataFormats_FWLite_MultiChainEvent_h
#define DataFormats_FWLite_MultiChainEvent_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     MultiChainEvent
//
/**\class MultiChainEvent MultiChainEvent.h DataFormats/FWLite/interface/MultiChainEvent.h

 Description: FWLite chain event that is aware of two files at once

 Usage:
    <usage>

*/
//
// Original Author:  Salvatore Rappoccio
//         Created:  Thu Jul  9 22:05:56 CDT 2009
//
#if !defined(__CINT__) && !defined(__MAKECINT__)
// system include files
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

// user include files
#include "DataFormats/FWLite/interface/EventBase.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "FWCore/Utilities/interface/HideStdSharedPtrFromRoot.h"

// forward declarations
namespace edm {
  class EDProduct;
  class ProductRegistry;
  class ProcessHistory;
  class BranchDescription;
  class EDProductGetter;
  class EventAux;
  class TriggerResults;
  class TriggerNames;
  class TriggerResultsByName;
}

namespace fwlite {

  namespace internal {
    class MultiProductGetter;
  }

class MultiChainEvent: public EventBase
{

   public:

      typedef std::map<edm::EventID, Long64_t>      sec_file_index_map;
      typedef std::pair<edm::EventID, edm::EventID> event_id_range;
      typedef std::map<event_id_range, Long64_t>    sec_file_range_index_map;

      MultiChainEvent(std::vector<std::string> const& iFileNames1,
		      std::vector<std::string> const& iFileNames2,
		      bool useSecFileMapSorted = false);
      virtual ~MultiChainEvent();

      const MultiChainEvent& operator++();

      ///Go to the event at index iIndex
      bool to(Long64_t iIndex);

      //If lumi is non-zero, Go to event by Run, Lumi, and Event number
      //If lumi is zero, Go to event by Run and Event number
      bool to(edm::EventID id);
      bool to(edm::RunNumber_t run, edm::EventNumber_t event);
      bool to(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event);

      // Go to the very first Event.
      const MultiChainEvent& toBegin();

      // ---------- const member functions ---------------------
      virtual std::string const getBranchNameFor(std::type_info const&,
                                                 char const*,
                                                 char const*,
                                                 char const*) const;

      /** This function should only be called by fwlite::Handle<>*/
      virtual bool getByLabel(std::type_info const&, char const*, char const*, char const*, void*) const;
      //void getByBranchName(std::type_info const&, char const*, void*&) const;

      bool isValid() const;
      operator bool() const;
      bool atEnd() const;

      Long64_t size() const;

      virtual edm::EventAuxiliary const& eventAuxiliary() const;

      std::vector<edm::BranchDescription> const& getBranchDescriptions() const;
      std::vector<std::string> const& getProcessHistory() const;
      edm::ProcessHistory const& processHistory() const;
      TFile* getTFile() const {
        return event1_->getTFile();
      }
      TFile* getTFileSec() const {
        return event2_->getTFile();
      }

      Long64_t eventIndex()    const { return event1_->eventIndex(); }
      Long64_t eventIndexSec() const { return event2_->eventIndex(); }

      fwlite::LuminosityBlock const& getLuminosityBlock() {
        return event1_->getLuminosityBlock();
      }

      fwlite::Run const& getRun() {
        return event1_->getRun();
      }


      virtual Long64_t fileIndex()          const
      { return event1_->eventIndex(); }
      virtual Long64_t secondaryFileIndex() const
      { return event2_->eventIndex(); }

      virtual edm::TriggerNames const& triggerNames(edm::TriggerResults const& triggerResults) const;
      virtual edm::TriggerResultsByName triggerResultsByName(std::string const& process) const;

      // ---------- static member functions --------------------
      static void throwProductNotFoundException(std::type_info const&, char const*, char const*, char const*);

      // return the two chain events
      ChainEvent const* primary  () const { return &*event1_;}
      ChainEvent const* secondary() const { return &*event2_;}

      // ---------- member functions ---------------------------

      edm::EDProduct const* getByProductID(edm::ProductID const&) const;


   private:

      MultiChainEvent(Event const&); // stop default

      const MultiChainEvent& operator= (Event const&); // stop default

      ///Go to the event from secondary files at index iIndex
      bool toSec(Long64_t iIndex);

      //Go to event from secondary files by Run, Lumi (if non-zero), and  Event number
      bool toSec(const edm::EventID &id);
      bool toSec(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event);
      bool toSec(edm::RunNumber_t run, edm::EventNumber_t event);


      // ---------- member data --------------------------------

      std::shared_ptr<ChainEvent> event1_;  // primary files
      std::shared_ptr<ChainEvent> event2_;  // secondary files
      std::shared_ptr<internal::MultiProductGetter> getter_;

      // speed up secondary file access with a (run range)_1 ---> index_2 map,
      // when the files are sorted by run,event within the file.
      // in this case, it is sufficient to store only a run-range to index mapping.
      // with this, the solution becomes more performant.
      bool                     useSecFileMapSorted_;
      sec_file_range_index_map secFileMapSorted_;
};

}
#endif /*__CINT__ */
#endif
