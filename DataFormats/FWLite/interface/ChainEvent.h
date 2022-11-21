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
// system include files
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

// user include files
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/EventBase.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
namespace edm {
  class WrapperBase;
  class ProductRegistry;
  class ProcessHistory;
  class BranchDescription;
  class EDProductGetter;
  class EventAux;
  class TriggerResults;
  class TriggerNames;
  class TriggerResultsByName;
}  // namespace edm

namespace fwlite {

  class ChainEvent : public EventBase {
  public:
    ChainEvent(std::vector<std::string> const& iFileNames);
    ~ChainEvent() override;

    ChainEvent const& operator++() override;

    ///Go to the event at index iIndex
    bool to(Long64_t iIndex);

    // If lumi is non-zero, go to event by Run, Lumi and Event number
    // If lumi is 0, go to event by Run and Event number only.
    bool to(const edm::EventID& id);
    bool to(edm::RunNumber_t run, edm::EventNumber_t event);
    bool to(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event);

    // Go to the very first Event.
    ChainEvent const& toBegin() override;

    // ---------- const member functions ---------------------
    std::string const getBranchNameFor(std::type_info const&, char const*, char const*, char const*) const override;
    template <typename T>
    edm::EDGetTokenT<T> consumes(edm::InputTag const& iTag) const {
      return event_->consumes<T>(iTag);
    }
    using fwlite::EventBase::getByLabel;

    // This function should only be called by fwlite::Handle<>
    bool getByLabel(std::type_info const&, char const*, char const*, char const*, void*) const override;
    //void getByBranchName(std::type_info const&, char const*, void*&) const;

    bool isValid() const;
    operator bool() const;
    bool atEnd() const override;

    Long64_t size() const;

    edm::EventAuxiliary const& eventAuxiliary() const override;

    std::vector<edm::BranchDescription> const& getBranchDescriptions() const;
    std::vector<std::string> const& getProcessHistory() const;
    edm::ProcessHistory const& processHistory() const override;
    TFile* getTFile() const { return event_->getTFile(); }

    // These functions return the index of the file that the current event
    // resides in. Note that the file index is based on the vector of files
    // which were actually opened, not the vector of input files in the
    // constructor. These two may differ in the case some input files contain
    // 0 events. To get the path of the file where the current event resides
    // in, fwlite::ChainEvent::getTFile()->GetPath() is preferred.
    Long64_t eventIndex() const { return eventIndex_; }
    Long64_t fileIndex() const override { return eventIndex_; }

    void setGetter(std::shared_ptr<edm::EDProductGetter const> getter) { event_->setGetter(getter); }

    Event const* event() const { return &*event_; }

    edm::TriggerNames const& triggerNames(edm::TriggerResults const& triggerResults) const override;
    void fillParameterSetRegistry() const;
    edm::TriggerResultsByName triggerResultsByName(edm::TriggerResults const& triggerResults) const override;

    edm::ParameterSet const* parameterSet(edm::ParameterSetID const& psID) const override;

    // ---------- static member functions --------------------
    static void throwProductNotFoundException(std::type_info const&, char const*, char const*, char const*);

    // ---------- member functions ---------------------------

    edm::WrapperBase const* getByProductID(edm::ProductID const&) const override;
    std::optional<std::tuple<edm::WrapperBase const*, unsigned int>> getThinnedProduct(edm::ProductID const& pid,
                                                                                       unsigned int key) const;

    void getThinnedProducts(edm::ProductID const& pid,
                            std::vector<edm::WrapperBase const*>& foundContainers,
                            std::vector<unsigned int>& keys) const;

    edm::OptionalThinnedKey getThinnedKeyFrom(edm::ProductID const& parent,
                                              unsigned int key,
                                              edm::ProductID const& thinned) const;

    fwlite::LuminosityBlock const& getLuminosityBlock();
    fwlite::Run const& getRun();

  private:
    bool getByTokenImp(edm::EDGetToken, edm::WrapperBase const*&) const override;

    friend class MultiChainEvent;

    ChainEvent(Event const&);  // stop default

    ChainEvent const& operator=(Event const&);  // stop default

    void findSizes();
    void switchToFile(Long64_t);
    // ---------- member data --------------------------------
    std::vector<std::string> fileNames_;
    edm::propagate_const<std::shared_ptr<TFile>> file_;
    edm::propagate_const<std::shared_ptr<Event>> event_;
    Long64_t eventIndex_;
    std::vector<Long64_t> accumulatedSize_;
    edm::propagate_const<std::shared_ptr<edm::EDProductGetter>> getter_;
  };

}  // namespace fwlite
#endif
