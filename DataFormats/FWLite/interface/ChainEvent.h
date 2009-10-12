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
// $Id: ChainEvent.h,v 1.6 2009/07/12 05:09:08 srappocc Exp $
//
#if !defined(__CINT__) && !defined(__MAKECINT__)
// system include files
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>

// user include files
#include "DataFormats/FWLite/interface/Event.h"

// forward declarations
namespace edm {
  class EDProduct;
  class ProductRegistry;
  class BranchDescription;
  class EDProductGetter;
  class EventAux;
}

namespace fwlite {
class ChainEvent
{

   public:

      ChainEvent(const std::vector<std::string>& iFileNames);
      virtual ~ChainEvent();

      const ChainEvent& operator++();

      ///Go to the event at index iIndex
      const ChainEvent& to(Long64_t iIndex);

      //Go to event by Run & Event number
      const ChainEvent & to(edm::EventID id);
      const ChainEvent & to(edm::RunNumber_t run, edm::EventNumber_t event);

      /** Go to the very first Event*/
      const ChainEvent& toBegin();
      
      // ---------- const member functions ---------------------
      const std::string getBranchNameFor(const std::type_info&, const char*, const char*, const char*) const;

      /** This function should only be called by fwlite::Handle<>*/
      bool getByLabel(const std::type_info&, const char*, const char*, const char*, void*) const;
      //void getByBranchName(const std::type_info&, const char*, void*&) const;

      bool isValid() const;
      operator bool () const;
      bool atEnd() const;
      
      Long64_t size() const;

      edm::EventID id() const;
      const edm::Timestamp& time() const;

      const std::vector<edm::BranchDescription>& getBranchDescriptions() const;
      const std::vector<std::string>& getProcessHistory() const;
      TFile* getTFile() const {
        return event_->getTFile();
      }

      Long64_t eventIndex() const { return eventIndex_; }

      void setGetter( boost::shared_ptr<edm::EDProductGetter> getter ){
	event_->setGetter( getter );
      }

      Event const * event() const { return &*event_; }

      // ---------- static member functions --------------------
      static void throwProductNotFoundException(const std::type_info&, const char*, const char*, const char*);

      // ---------- member functions ---------------------------

      edm::EDProduct const* getByProductID(edm::ProductID const&) const;

   private:

      friend class MultiChainEvent;

      ChainEvent(const Event&); // stop default

      const ChainEvent& operator=(const Event&); // stop default

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
