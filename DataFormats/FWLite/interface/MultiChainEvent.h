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
// $Id: MultiChainEvent.h,v 1.7 2009/09/04 21:34:19 wdd Exp $
//
#if !defined(__CINT__) && !defined(__MAKECINT__)
// system include files
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>

// user include files
#include "DataFormats/FWLite/interface/EventBase.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"

// forward declarations
namespace edm {
  class EDProduct;
  class ProductRegistry;
  class BranchDescription;
  class EDProductGetter;
  class EventAux;
  class TriggerResults;
}

namespace fwlite {
  class TriggerNames;

  namespace internal {
    class MultiProductGetter;
  }

class MultiChainEvent: public EventBase
{

   public:
      MultiChainEvent(const std::vector<std::string>& iFileNames1, 
		      const std::vector<std::string>& iFileNames2);
      virtual ~MultiChainEvent();

      const MultiChainEvent& operator++();

      ///Go to the event at index iIndex
      const MultiChainEvent& to(Long64_t iIndex);

      //Go to event by Run & Event number
      const MultiChainEvent & to(edm::EventID id);
      const MultiChainEvent & to(edm::RunNumber_t run, edm::EventNumber_t event);

      // Go to the very first Event. 
      const MultiChainEvent& toBegin();
      
      // ---------- const member functions ---------------------
      virtual const std::string getBranchNameFor(const std::type_info&, 
                                                 const char*, 
                                                 const char*, 
                                                 const char*) const;

      /** This function should only be called by fwlite::Handle<>*/
      virtual bool getByLabel(const std::type_info&, const char*, 
                              const char*, const char*, void*) const;
      //void getByBranchName(const std::type_info&, const char*, void*&) const;

      bool isValid() const;
      operator bool () const;
      bool atEnd() const;
      
      Long64_t size() const;

      virtual edm::EventAuxiliary const& eventAuxiliary() const;

      const std::vector<edm::BranchDescription>& getBranchDescriptions() const;
      const std::vector<std::string>& getProcessHistory() const;
      TFile* getTFile() const {
        return event1_->getTFile();
      }
      TFile* getTFileSec() const {
        return event2_->getTFile();
      }

      Long64_t eventIndex()    const { return event1_->eventIndex(); }
      Long64_t eventIndexSec() const { return event2_->eventIndex(); }
      virtual Long64_t fileIndex()          const 
      { return event1_->eventIndex(); }
      virtual Long64_t secondaryFileIndex() const 
      { return event2_->eventIndex(); }

      virtual TriggerNames const& triggerNames(edm::TriggerResults const& triggerResults);

      // ---------- static member functions --------------------
      static void throwProductNotFoundException(const std::type_info&, const char*, const char*, const char*);

      // return the two chain events
      ChainEvent const * primary  () const { return &*event1_;}
      ChainEvent const * secondary() const { return &*event2_;}

      // ---------- member functions ---------------------------

      edm::EDProduct const* getByProductID(edm::ProductID const&) const;


   private:

      MultiChainEvent(const Event&); // stop default

      const MultiChainEvent& operator=(const Event&); // stop default

      ///Go to the event from secondary files at index iIndex
      const MultiChainEvent& toSec(Long64_t iIndex);

      //Go to event from secondary files by Run & Event number
      const MultiChainEvent & toSec(edm::EventID id);
      const MultiChainEvent & toSec(edm::RunNumber_t run, edm::EventNumber_t event);


      // ---------- member data --------------------------------

      boost::shared_ptr<ChainEvent> event1_;  // primary files
      boost::shared_ptr<ChainEvent> event2_;  // secondary files
      boost::shared_ptr<internal::MultiProductGetter> getter_;
};

}
#endif /*__CINT__ */
#endif
