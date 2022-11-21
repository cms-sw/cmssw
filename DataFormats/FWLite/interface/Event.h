#ifndef DataFormats_FWLite_Event_h
#define DataFormats_FWLite_Event_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     Event
//
/**\class Event Event.h DataFormats/FWLite/interface/Event.h

   Description: Provide event data access in FWLite

   Usage:
   This class is meant to allow one to loop over all events in a TFile and then
 read the data in an Event in a manner analogous to how data is read in the full framework.
 A typical use would be
 \code
 TFile f("foo.root");
 fwlite::Event ev(&f);
 for(ev.toBeing(); ! ev.atEnd(); ++ev) {
    fwlite::Handle<std::vector<Foo> > foos;
    foos.getByLabel(ev, "myFoos");
 }
 \endcode
 The above example will work for both ROOT and compiled code. However, it is possible to exactly
 match the full framework if you only intend to compile your code.  In that case the access
 would look like

 \code
 TFile f("foo.root");
 fwlite::Event ev(&f);

 edm::InputTag fooTag("myFoos");
 for(ev.toBeing(); ! ev.atEnd(); ++ev) {
    edm::Handle<std::vector<Foo> > foos;
    ev.getByLabel(fooTag, foos);
 }
 \endcode

 NOTE: This class is not safe to use across threads.
*/
//
// Original Author:  Chris Jones
//         Created:  Tue May  8 15:01:20 EDT 2007
//
// system include files
#include <typeinfo>
#include <map>
#include <vector>
#include <memory>
#include <cstring>
#include <string>
#include <functional>

#include "Rtypes.h"

// user include files
#include "DataFormats/FWLite/interface/EventBase.h"
#include "DataFormats/FWLite/interface/EntryFinder.h"
#include "DataFormats/FWLite/interface/LuminosityBlock.h"
#include "DataFormats/FWLite/interface/Run.h"
#include "DataFormats/FWLite/interface/InternalDataKey.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/EventProcessHistoryID.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

// forward declarations
namespace edm {
  class WrapperBase;
  class ProductRegistry;
  class BranchDescription;
  class EDProductGetter;
  class EventAux;
  class Timestamp;
  class TriggerResults;
  class TriggerNames;
  class TriggerResultsByName;
}  // namespace edm
class TCut;

namespace fwlite {
  class BranchMapReader;
  class HistoryGetterBase;
  class DataGetterHelper;
  class RunFactory;
  class ChainEvent;
  class MultiChainEvent;

  class Event : public EventBase {
  public:
    friend class ChainEvent;
    friend class MultiChainEvent;

    // NOTE: Does NOT take ownership so iFile must remain around
    // at least as long as Event.
    // useCache and baFunc (branch-access-function) are passed to
    // DataGetterHelper and help with external management of TTreeCache
    // associated with the file. By default useCache is true and internal
    // DataGetterHelper caching is enabled. When user sets useCache to
    // false no cache is created unless user attaches and controls it
    // himself.
    Event(
        TFile* iFile, bool useCache = true, std::function<void(TBranch const&)> baFunc = [](TBranch const&) {});

    Event(Event const&) = delete;  // stop default

    Event const& operator=(Event const&) = delete;  // stop default

    ~Event() override;

    ///Advance to next event in the TFile
    Event const& operator++() override;

    ///Find index of given event-id
    Long64_t indexFromEventId(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event);

    ///Go to the event at index iIndex
    bool to(Long64_t iIndex);

    ///Go to event by Run & Event number
    bool to(const edm::EventID& id);
    bool to(edm::RunNumber_t run, edm::EventNumber_t event);
    bool to(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event);

    /// Go to the very first Event.
    Event const& toBegin() override;

    // ---------- const member functions ---------------------
    ///Return the branch name in the TFile which contains the data
    std::string const getBranchNameFor(std::type_info const&,
                                       char const* iModuleLabel,
                                       char const* iProductInstanceLabel,
                                       char const* iProcessName) const override;

    template <typename T>
    edm::EDGetTokenT<T> consumes(edm::InputTag const& iTag) const {
      auto bid =
          dataHelper_.getBranchIDFor(typeid(T), iTag.label().c_str(), iTag.instance().c_str(), iTag.process().c_str());
      if (bid) {
        return this->makeTokenUsing<T>(bid.value().id());
      }
      return {};
    }
    using fwlite::EventBase::getByLabel;
    /// This function should only be called by fwlite::Handle<>
    bool getByLabel(std::type_info const&, char const*, char const*, char const*, void*) const override;
    //void getByBranchName(std::type_info const&, char const*, void*&) const;

    ///Properly setup for edm::Ref, etc and then call TTree method
    void draw(Option_t* opt);
    Long64_t draw(char const* varexp,
                  const TCut& selection,
                  Option_t* option = "",
                  Long64_t nentries = 1000000000,
                  Long64_t firstentry = 0);
    Long64_t draw(char const* varexp,
                  char const* selection,
                  Option_t* option = "",
                  Long64_t nentries = 1000000000,
                  Long64_t firstentry = 0);
    Long64_t scan(char const* varexp = "",
                  char const* selection = "",
                  Option_t* option = "",
                  Long64_t nentries = 1000000000,
                  Long64_t firstentry = 0);

    bool isValid() const;
    operator bool() const;
    bool atEnd() const override;

    ///Returns number of events in the file
    Long64_t size() const;

    edm::EventAuxiliary const& eventAuxiliary() const override;

    std::vector<edm::BranchDescription> const& getBranchDescriptions() const {
      return branchMap_.getBranchDescriptions();
    }
    std::vector<std::string> const& getProcessHistory() const;
    TFile* getTFile() const { return branchMap_.getFile(); }

    edm::ParameterSet const* parameterSet(edm::ParameterSetID const& psID) const override;

    edm::WrapperBase const* getByProductID(edm::ProductID const&) const override;
    std::optional<std::tuple<edm::WrapperBase const*, unsigned int>> getThinnedProduct(edm::ProductID const& pid,
                                                                                       unsigned int key) const;
    void getThinnedProducts(edm::ProductID const& pid,
                            std::vector<edm::WrapperBase const*>& foundContainers,
                            std::vector<unsigned int>& keys) const;
    edm::OptionalThinnedKey getThinnedKeyFrom(edm::ProductID const& parent,
                                              unsigned int key,
                                              edm::ProductID const& thinned) const;

    edm::TriggerNames const& triggerNames(edm::TriggerResults const& triggerResults) const override;

    edm::TriggerResultsByName triggerResultsByName(edm::TriggerResults const& triggerResults) const override;

    edm::ProcessHistory const& processHistory() const override { return history(); }

    fwlite::LuminosityBlock const& getLuminosityBlock() const;
    fwlite::Run const& getRun() const;

    // ---------- static member functions --------------------
    static void throwProductNotFoundException(std::type_info const&, char const*, char const*, char const*);

  private:
    bool getByTokenImp(edm::EDGetToken, edm::WrapperBase const*&) const override;
    friend class internal::ProductGetter;
    friend class ChainEvent;
    friend class EventHistoryGetter;

    edm::ProcessHistory const& history() const;
    void updateAux(Long_t eventIndex) const;
    void fillParameterSetRegistry() const;
    void setGetter(std::shared_ptr<edm::EDProductGetter const> getter) { return dataHelper_.setGetter(getter); }

    // ---------- member data --------------------------------
    //This class is not inteded to be used across different threads
    CMS_SA_ALLOW mutable TFile* file_;
    // TTree* eventTree_;
    TTree* eventHistoryTree_;
    // Long64_t eventIndex_;
    CMS_SA_ALLOW mutable std::shared_ptr<fwlite::LuminosityBlock> lumi_;
    CMS_SA_ALLOW mutable std::shared_ptr<fwlite::Run> run_;
    CMS_SA_ALLOW mutable fwlite::BranchMapReader branchMap_;

    //takes ownership of the strings used by the DataKey keys in data_
    CMS_SA_ALLOW mutable std::vector<char const*> labels_;
    CMS_SA_ALLOW mutable edm::ProcessHistoryMap historyMap_;
    CMS_SA_ALLOW mutable std::vector<edm::EventProcessHistoryID> eventProcessHistoryIDs_;
    CMS_SA_ALLOW mutable std::vector<std::string> procHistoryNames_;
    CMS_SA_ALLOW mutable edm::EventAuxiliary aux_;
    CMS_SA_ALLOW mutable EntryFinder entryFinder_;
    edm::EventAuxiliary const* pAux_;
    edm::EventAux const* pOldAux_;
    TBranch* auxBranch_;
    int fileVersion_;
    CMS_SA_ALLOW mutable bool parameterSetRegistryFilled_;

    fwlite::DataGetterHelper dataHelper_;
    CMS_SA_ALLOW mutable std::shared_ptr<RunFactory> runFactory_;
  };

}  // namespace fwlite
#endif
