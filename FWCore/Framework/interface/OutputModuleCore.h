#ifndef FWCore_Framework_one_OutputModuleCore_h
#define FWCore_Framework_one_OutputModuleCore_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     OutputModuleCore
//
/**\class OutputModuleCore OutputModuleCore.h "FWCore/Framework/interface/one/OutputModuleCore.h"

 Description: Base class for all 'one' OutputModules

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 31 Jul 2013 15:37:16 GMT
//

// system include files
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <atomic>
#include <set>

// user include files
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/SelectedProducts.h"

#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Common/interface/OutputProcessBlockHelper.h"
#include "FWCore/Framework/interface/TriggerResultsBasedEventSelector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Framework/interface/ProductSelector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/getAllTriggerNames.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
namespace edm {

  class MergeableRunProductMetadata;
  class ModuleCallingContext;
  class PreallocationConfiguration;
  class ActivityRegistry;
  class ThinnedAssociationsHelper;

  template <typename T>
  class OutputModuleCommunicatorT;

  namespace maker {
    template <typename T>
    class ModuleHolderT;
  }

  namespace core {

    class OutputModuleCore : public EDConsumerBase {
    public:
      template <typename U>
      friend class edm::maker::ModuleHolderT;
      template <typename T>
      friend class ::edm::WorkerT;
      template <typename T>
      friend class ::edm::OutputModuleCommunicatorT;
      typedef OutputModuleCore ModuleType;

      explicit OutputModuleCore(ParameterSet const& pset);
      ~OutputModuleCore() override;

      OutputModuleCore(OutputModuleCore const&) = delete;             // Disallow copying and moving
      OutputModuleCore& operator=(OutputModuleCore const&) = delete;  // Disallow copying and moving

      /// Accessor for maximum number of events to be written.
      /// -1 is used for unlimited.
      int maxEvents() const { return maxEvents_; }

      /// Accessor for remaining number of events to be written.
      /// -1 is used for unlimited.
      int remainingEvents() const { return remainingEvents_; }

      bool selected(BranchDescription const& desc) const;

      void selectProducts(ProductRegistry const& preg, ThinnedAssociationsHelper const&, ProcessBlockHelperBase const&);
      std::string const& processName() const { return process_name_; }
      SelectedProductsForBranchType const& keptProducts() const { return keptProducts_; }
      std::array<bool, NumBranchTypes> const& hasNewlyDroppedBranch() const { return hasNewlyDroppedBranch_; }

      static void fillDescription(
          ParameterSetDescription& desc,
          std::vector<std::string> const& iDefaultOutputCommands = ProductSelectorRules::defaultSelectionStrings());
      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      static const std::string& baseType();
      static void prevalidate(ConfigurationDescriptions&);

      bool wantAllEvents() const { return wantAllEvents_; }

      BranchIDLists const* branchIDLists() const;

      OutputProcessBlockHelper const& outputProcessBlockHelper() const { return outputProcessBlockHelper_; }

      ThinnedAssociationsHelper const* thinnedAssociationsHelper() const;

      const ModuleDescription& moduleDescription() const { return moduleDescription_; }

      void callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func) {
        callWhenNewProductsRegistered_ = func;
      }

    protected:
      ModuleDescription const& description() const;

      ParameterSetID selectorConfig() const { return selector_config_id_; }

      void doPreallocate_(PreallocationConfiguration const&);
      virtual void preallocLumis(unsigned int);

      void doBeginJob_();
      void doEndJob();
      bool doEvent_(EventTransitionInfo const&, ActivityRegistry*, ModuleCallingContext const*);
      void doBeginProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) {}
      void doAccessInputProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) {}
      void doEndProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) {}
      bool doBeginRun(RunTransitionInfo const&, ModuleCallingContext const*);
      bool doEndRun(RunTransitionInfo const&, ModuleCallingContext const*);
      bool doBeginLuminosityBlock(LumiTransitionInfo const&, ModuleCallingContext const*);
      bool doEndLuminosityBlock(LumiTransitionInfo const&, ModuleCallingContext const*);

      void setEventSelectionInfo(
          std::map<std::string, std::vector<std::pair<std::string, int>>> const& outputModulePathPositions,
          bool anyProductProduced);

      void configure(OutputModuleDescription const& desc);

      std::map<BranchID::value_type, BranchID::value_type> const& droppedBranchIDToKeptBranchID() {
        return droppedBranchIDToKeptBranchID_;
      }

      //inheriting classes decrement this in doEvent in a manner that will be thread-safe for that class
      std::atomic<int> remainingEvents_;

    private:
      int maxEvents_;

      // TODO: Give OutputModule
      // an interface (protected?) that supplies client code with the
      // needed functionality *without* giving away implementation
      // details ... don't just return a reference to keptProducts_, because
      // we are looking to have the flexibility to change the
      // implementation of keptProducts_ without modifying clients. When this
      // change is made, we'll have a one-time-only task of modifying
      // clients (classes derived from OutputModule) to use the
      // newly-introduced interface.
      // TODO: Consider using shared pointers here?

      // keptProducts_ are pointers to the BranchDescription objects describing
      // the branches we are to write.
      //
      // We do not own the BranchDescriptions to which we point.
      SelectedProductsForBranchType keptProducts_;
      std::array<bool, NumBranchTypes> hasNewlyDroppedBranch_;

      std::string process_name_;
      ProductSelectorRules productSelectorRules_;
      ProductSelector productSelector_;
      ModuleDescription moduleDescription_;

      bool wantAllEvents_;
      std::vector<detail::TriggerResultsBasedEventSelector> selectors_;
      ParameterSet selectEvents_;
      std::vector<EDGetToken> tokensForEndPaths_;  //needed for FinalPath
      bool onFinalPath_ = false;
      // ID of the ParameterSet that configured the event selector
      // subsystem.
      ParameterSetID selector_config_id_;

      // needed because of possible EDAliases.
      // filled in only if key and value are different.
      std::map<BranchID::value_type, BranchID::value_type> droppedBranchIDToKeptBranchID_;
      edm::propagate_const<std::unique_ptr<BranchIDLists>> branchIDLists_;
      BranchIDLists const* origBranchIDLists_;

      edm::propagate_const<std::unique_ptr<ThinnedAssociationsHelper>> thinnedAssociationsHelper_;
      std::map<BranchID, bool> keepAssociation_;

      OutputProcessBlockHelper outputProcessBlockHelper_;

      std::function<void(BranchDescription const&)> callWhenNewProductsRegistered_;

      //------------------------------------------------------------------
      // private member functions
      //------------------------------------------------------------------
      void updateBranchIDListsWithKeptAliases();

      void doWriteProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*);
      void doWriteRun(RunPrincipal const& rp, ModuleCallingContext const*, MergeableRunProductMetadata const*);
      void doWriteLuminosityBlock(LuminosityBlockPrincipal const& lbp, ModuleCallingContext const*);
      void doOpenFile(FileBlock const& fb);
      void doRespondToOpenInputFile(FileBlock const& fb);
      void doRespondToCloseInputFile(FileBlock const& fb);
      void doRespondToCloseOutputFile() {}
      void doRegisterThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) {}

      /// Tell the OutputModule that is must end the current file.
      void doCloseFile();

      void registerProductsAndCallbacks(OutputModuleCore const*, ProductRegistry*);

      bool needToRunSelection() const;
      std::vector<ProductResolverIndexAndSkipBit> productsUsedBySelection() const;
      bool prePrefetchSelection(StreamID id, EventPrincipal const&, ModuleCallingContext const*);

      // Do the end-of-file tasks; this is only called internally, after
      // the appropriate tests have been done.
      virtual void reallyCloseFile();

      /// Ask the OutputModule if we should end the current file.
      virtual bool shouldWeCloseFile() const { return false; }

      virtual void write(EventForOutput const&) = 0;

      virtual void beginJob() {}
      virtual void endJob() {}
      virtual void writeLuminosityBlock(LuminosityBlockForOutput const&) = 0;
      virtual void writeRun(RunForOutput const&) = 0;
      virtual void writeProcessBlock(ProcessBlockForOutput const&) {}
      virtual void openFile(FileBlock const&) {}
      virtual bool isFileOpen() const { return true; }

      virtual void doBeginRun_(RunForOutput const&) {}
      virtual void doEndRun_(RunForOutput const&) {}
      virtual void doBeginLuminosityBlock_(LuminosityBlockForOutput const&) {}
      virtual void doEndLuminosityBlock_(LuminosityBlockForOutput const&) {}
      virtual void doRespondToOpenInputFile_(FileBlock const&) {}
      virtual void doRespondToCloseInputFile_(FileBlock const&) {}

      virtual void setProcessesWithSelectedMergeableRunProducts(std::set<std::string> const&) {}

      bool hasAccumulator() const { return false; }

      void keepThisBranch(BranchDescription const& desc,
                          std::map<BranchID, BranchDescription const*>& trueBranchIDToKeptBranchDesc,
                          std::set<BranchID>& keptProductsInEvent);

      void setModuleDescription(ModuleDescription const& md) { moduleDescription_ = md; }

      bool limitReached() const { return remainingEvents_ == 0; }
    };
  }  // namespace core
}  // namespace edm
#endif
