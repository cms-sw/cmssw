#ifndef DataFormats_FWLite_LuminosityBlock_h
#define DataFormats_FWLite_LuminosityBlock_h
// -*- C++ -*-
//
// Package:     FWLite/DataFormats
// Class  :     LuminosityBlock
//
/**\class LuminosityBlock LuminosityBlock.h DataFormats/FWLite/interface/LuminosityBlock.h

   Description: <one line class summary>

   Usage:
   This class is not safe to use across threads

*/
//
// Original Author:  Eric Vaandering
//         Created:  Wed Jan 13 15:01:20 EDT 2007
//
// system include files
#include <typeinfo>
#include <map>
#include <vector>
#include <memory>
#include <cstring>

#include "Rtypes.h"

// user include files
#include "DataFormats/FWLite/interface/Run.h"
#include "DataFormats/FWLite/interface/LuminosityBlockBase.h"
#include "DataFormats/FWLite/interface/InternalDataKey.h"
#include "DataFormats/FWLite/interface/EntryFinder.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

// forward declarations
namespace edm {
  class WrapperBase;
  class ProductRegistry;
  class BranchDescription;
  class EDProductGetter;
  class LuminosityBlockAux;
  class Timestamp;
  class TriggerResults;
  class TriggerNames;
}  // namespace edm

namespace fwlite {
  class Event;
  class BranchMapReader;
  class HistoryGetterBase;
  class DataGetterHelper;
  class RunFactory;

  class LuminosityBlock : public LuminosityBlockBase {
  public:
    // NOTE: Does NOT take ownership so iFile must remain around
    // at least as long as LuminosityBlock
    LuminosityBlock(TFile* iFile);
    LuminosityBlock(std::shared_ptr<BranchMapReader> branchMap, std::shared_ptr<RunFactory> runFactory);
    ~LuminosityBlock() override;

    const LuminosityBlock& operator++() override;

    /// Go to event by Run & LuminosityBlock number
    bool to(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi);

    // Go to the very first Event.
    const LuminosityBlock& toBegin() override;

    // ---------- const member functions ---------------------
    virtual std::string const getBranchNameFor(std::type_info const&, char const*, char const*, char const*) const;

    // This function should only be called by fwlite::Handle<>
    using fwlite::LuminosityBlockBase::getByLabel;
    bool getByLabel(std::type_info const&, char const*, char const*, char const*, void*) const override;
    //void getByBranchName(std::type_info const&, char const*, void*&) const;

    bool isValid() const;
    operator bool() const;
    bool atEnd() const override;

    Long64_t size() const;

    edm::LuminosityBlockAuxiliary const& luminosityBlockAuxiliary() const override;

    std::vector<edm::BranchDescription> const& getBranchDescriptions() const {
      return branchMap_->getBranchDescriptions();
    }

    //       void setGetter(//Copy from Event if needed

    edm::WrapperBase const* getByProductID(edm::ProductID const&) const;

    // ---------- static member functions --------------------
    static void throwProductNotFoundException(std::type_info const&, char const*, char const*, char const*);

    // ---------- member functions ---------------------------
    fwlite::Run const& getRun() const;

  private:
    friend class internal::ProductGetter;
    friend class LumiHistoryGetter;

    LuminosityBlock(const LuminosityBlock&) = delete;  // stop default

    const LuminosityBlock& operator=(const LuminosityBlock&) = delete;  // stop default

    const edm::ProcessHistory& history() const;
    void updateAux(Long_t lumiIndex) const;

    // ---------- member data --------------------------------
    //This class is not inteded to be used across different threads
    CMS_SA_ALLOW mutable std::shared_ptr<BranchMapReader> branchMap_;

    CMS_SA_ALLOW mutable std::shared_ptr<fwlite::Run> run_;

    //takes ownership of the strings used by the DataKey keys in data_
    CMS_SA_ALLOW mutable std::vector<char const*> labels_;
    CMS_SA_ALLOW mutable edm::ProcessHistoryMap historyMap_;
    CMS_SA_ALLOW mutable std::vector<std::string> procHistoryNames_;
    CMS_SA_ALLOW mutable edm::LuminosityBlockAuxiliary aux_;
    CMS_SA_ALLOW mutable EntryFinder entryFinder_;
    edm::LuminosityBlockAuxiliary const* pAux_;
    edm::LuminosityBlockAux const* pOldAux_;
    TBranch* auxBranch_;
    int fileVersion_;

    DataGetterHelper dataHelper_;
    CMS_SA_ALLOW mutable std::shared_ptr<RunFactory> runFactory_;
  };

}  // namespace fwlite
#endif
