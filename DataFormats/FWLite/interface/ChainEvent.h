#ifndef DataFormats_FWLite_ChainEvent_h
#define DataFormats_FWLite_ChainEvent_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     ChainEvent
//
/**\class ChainEvent ChainEvent.h DataFormats/FWLite/interface/ChainEvent.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue May  8 15:01:20 EDT 2007
//
#if !defined(__CINT__) && !defined(__MAKECINT__)
// system include files
#include <string>
#include <typeinfo>
#include <vector>
#include "boost/shared_ptr.hpp"

// user include files
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/EventBase.h"

// forward declarations
namespace edm {
  class WrapperHolder;
  class ProductRegistry;
  class BranchDescription;
  class EDProductGetter;
  class EventAux;
  class TriggerResults;
  class TriggerNames;
  class TriggerResultsByName;
}

namespace fwlite {

   class ChainEvent : public EventBase
{

   public:

      ChainEvent(std::vector<std::string> const& iFileNames);
      virtual ~ChainEvent();

      ChainEvent const& operator++();

      ///Go to the event at index iIndex
      bool to(Long64_t iIndex);

      // If lumi is non-zero, go to event by Run, Lumi and Event number
      // If lumi is 0, go to event by Run and Event number only.
      bool to(const edm::EventID &id);
      bool to(edm::RunNumber_t run, edm::EventNumber_t event);
      bool to(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event);

      // Go to the very first Event.
      ChainEvent const& toBegin();

      // ---------- const member functions ---------------------
      virtual std::string const getBranchNameFor(std::type_info const&,
                                                 char const*,
                                                 char const*,
                                                 char const*) const;

      // This function should only be called by fwlite::Handle<>
      virtual bool getByLabel(std::type_info const&, char const*,
                              char const*, char const*, void*) const;
      //void getByBranchName(std::type_info const&, char const*, void*&) const;

      bool isValid() const;
      operator bool() const;
      virtual bool atEnd() const;

      Long64_t size() const;

      virtual edm::EventAuxiliary const& eventAuxiliary() const;

      std::vector<edm::BranchDescription> const& getBranchDescriptions() const;
      std::vector<std::string> const& getProcessHistory() const;
      TFile* getTFile() const {
        return event_->getTFile();
      }

      Long64_t eventIndex() const { return eventIndex_; }
      virtual Long64_t fileIndex() const { return eventIndex_; }

      void setGetter(boost::shared_ptr<edm::EDProductGetter> getter){
         event_->setGetter(getter);
      }

      Event const* event() const { return &*event_; }

      virtual edm::TriggerNames const& triggerNames(edm::TriggerResults const& triggerResults) const;
      void fillParameterSetRegistry() const;
      virtual edm::TriggerResultsByName triggerResultsByName(std::string const& process) const;

      // ---------- static member functions --------------------
      static void throwProductNotFoundException(std::type_info const&, char const*, char const*, char const*);

      // ---------- member functions ---------------------------

      edm::WrapperHolder getByProductID(edm::ProductID const&) const;
      fwlite::LuminosityBlock const& getLuminosityBlock();
      fwlite::Run             const& getRun();

   private:

      friend class MultiChainEvent;

      ChainEvent(Event const&); // stop default

      ChainEvent const& operator=(Event const&); // stop default

      void findSizes();
      void switchToFile(Long64_t);
      // ---------- member data --------------------------------
      std::vector<std::string> fileNames_;
      boost::shared_ptr<TFile> file_;
      boost::shared_ptr<Event> event_;
      Long64_t eventIndex_;
      std::vector<Long64_t> accumulatedSize_;
      boost::shared_ptr<edm::EDProductGetter> getter_;

};

}
#endif /*__CINT__ */
#endif
