#ifndef FWCore_Framework_SubProcess_h
#define FWCore_Framework_SubProcess_h

#include "DataFormats/Provenance/interface/BranchID.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/PathsAndConsumesOfModules.h"
#include "FWCore/Framework/src/PrincipalCache.h"
#include "FWCore/Framework/interface/ScheduleItems.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/TriggerResultsBasedEventSelector.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Framework/interface/ProductSelector.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include "DataFormats/Provenance/interface/SelectedProducts.h"

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace edm {
  class ActivityRegistry;
  class BranchDescription;
  class BranchIDListHelper;
  class EventSetupImpl;
  class HistoryAppender;
  class IOVSyncValue;
  class MergeableRunProductMetadata;
  class ParameterSet;
  class ProductRegistry;
  class PreallocationConfiguration;
  class ThinnedAssociationsHelper;
  class SubProcessParentageHelper;
  class WaitingTaskHolder;

  namespace eventsetup {
    class EventSetupsController;
  }
  class SubProcess : public EDConsumerBase {
  public:
    SubProcess(ParameterSet& parameterSet,
               ParameterSet const& topLevelParameterSet,
               std::shared_ptr<ProductRegistry const> parentProductRegistry,
               std::shared_ptr<BranchIDListHelper const> parentBranchIDListHelper,
               ThinnedAssociationsHelper const& parentThinnedAssociationsHelper,
               SubProcessParentageHelper const& parentSubProcessParentageHelper,
               eventsetup::EventSetupsController& esController,
               ActivityRegistry& parentActReg,
               ServiceToken const& token,
               serviceregistry::ServiceLegacy iLegacy,
               PreallocationConfiguration const& preallocConfig,
               ProcessContext const* parentProcessContext);

    ~SubProcess() override;

    SubProcess(SubProcess const&) = delete;             // Disallow copying
    SubProcess& operator=(SubProcess const&) = delete;  // Disallow copying
    SubProcess(SubProcess&&) = default;                 // Allow Moving
    SubProcess& operator=(SubProcess&&) = delete;       // Move not supported by PrincipalCache

    //From OutputModule
    void selectProducts(ProductRegistry const& preg,
                        ThinnedAssociationsHelper const& parentThinnedAssociationsHelper,
                        std::map<BranchID, bool>& keepAssociation);

    SelectedProductsForBranchType const& keptProducts() const { return keptProducts_; }

    void doBeginJob();
    void doEndJob();

    void doEventAsync(WaitingTaskHolder iHolder,
                      EventPrincipal const& principal,
                      std::vector<std::shared_ptr<const EventSetupImpl>> const*);

    void doBeginRunAsync(WaitingTaskHolder iHolder,
                         RunPrincipal const& principal,
                         IOVSyncValue const& ts,
                         std::vector<std::shared_ptr<const EventSetupImpl>> const*);

    void doEndRunAsync(WaitingTaskHolder iHolder,
                       RunPrincipal const& principal,
                       IOVSyncValue const& ts,
                       std::vector<std::shared_ptr<const EventSetupImpl>> const*,
                       bool cleaningUpAfterException);

    void doBeginLuminosityBlockAsync(WaitingTaskHolder iHolder,
                                     LuminosityBlockPrincipal const& principal,
                                     IOVSyncValue const& ts,
                                     std::vector<std::shared_ptr<const EventSetupImpl>> const*);

    void doEndLuminosityBlockAsync(WaitingTaskHolder iHolder,
                                   LuminosityBlockPrincipal const& principal,
                                   IOVSyncValue const& ts,
                                   std::vector<std::shared_ptr<const EventSetupImpl>> const*,
                                   bool cleaningUpAfterException);

    void doBeginStream(unsigned int);
    void doEndStream(unsigned int);
    void doStreamBeginRunAsync(WaitingTaskHolder iHolder,
                               unsigned int iID,
                               RunPrincipal const& principal,
                               IOVSyncValue const& ts,
                               std::vector<std::shared_ptr<const EventSetupImpl>> const*);

    void doStreamEndRunAsync(WaitingTaskHolder iHolder,
                             unsigned int iID,
                             RunPrincipal const& principal,
                             IOVSyncValue const& ts,
                             std::vector<std::shared_ptr<const EventSetupImpl>> const*,
                             bool cleaningUpAfterException);

    void doStreamBeginLuminosityBlockAsync(WaitingTaskHolder iHolder,
                                           unsigned int iID,
                                           LuminosityBlockPrincipal const& principal,
                                           IOVSyncValue const& ts,
                                           std::vector<std::shared_ptr<const EventSetupImpl>> const*);

    void doStreamEndLuminosityBlockAsync(WaitingTaskHolder iHolder,
                                         unsigned int iID,
                                         LuminosityBlockPrincipal const& principal,
                                         IOVSyncValue const& ts,
                                         std::vector<std::shared_ptr<const EventSetupImpl>> const*,
                                         bool cleaningUpAfterException);

    // Write the luminosity block
    void writeLumiAsync(WaitingTaskHolder, LuminosityBlockPrincipal&);

    void deleteLumiFromCache(LuminosityBlockPrincipal&);

    // Write the run
    void writeRunAsync(WaitingTaskHolder,
                       ProcessHistoryID const& parentPhID,
                       int runNumber,
                       MergeableRunProductMetadata const*);

    void deleteRunFromCache(ProcessHistoryID const& parentPhID, int runNumber);

    // Call closeFile() on all OutputModules.
    void closeOutputFiles() {
      ServiceRegistry::Operate operate(serviceToken_);
      schedule_->closeOutputFiles();
      for_all(subProcesses_, [](auto& subProcess) { subProcess.closeOutputFiles(); });
    }

    // Call openFiles() on all OutputModules
    void openOutputFiles(FileBlock& fb) {
      ServiceRegistry::Operate operate(serviceToken_);
      schedule_->openOutputFiles(fb);
      for_all(subProcesses_, [&fb](auto& subProcess) { subProcess.openOutputFiles(fb); });
    }

    void updateBranchIDListHelper(BranchIDLists const&);

    // Call respondToOpenInputFile() on all Modules
    void respondToOpenInputFile(FileBlock const& fb);

    // Call respondToCloseInputFile() on all Modules
    void respondToCloseInputFile(FileBlock const& fb) {
      ServiceRegistry::Operate operate(serviceToken_);
      schedule_->respondToCloseInputFile(fb);
      for_all(subProcesses_, [&fb](auto& subProcess) { subProcess.respondToCloseInputFile(fb); });
    }

    // Call shouldWeCloseFile() on all OutputModules.
    bool shouldWeCloseOutput() const {
      ServiceRegistry::Operate operate(serviceToken_);
      if (schedule_->shouldWeCloseOutput()) {
        return true;
      }
      for (auto const& subProcess : subProcesses_) {
        if (subProcess.shouldWeCloseOutput()) {
          return true;
        }
      }
      return false;
    }

    /// Return a vector allowing const access to all the ModuleDescriptions for this SubProcess

    /// *** N.B. *** Ownership of the ModuleDescriptions is *not*
    /// *** passed to the caller. Do not call delete on these
    /// *** pointers!
    std::vector<ModuleDescription const*> getAllModuleDescriptions() const;

    /// Return the number of events this SubProcess has tried to process
    /// (inclues both successes and failures, including failures due
    /// to exceptions during processing).
    int totalEvents() const { return schedule_->totalEvents(); }

    /// Return the number of events which have been passed by one or more trigger paths.
    int totalEventsPassed() const {
      ServiceRegistry::Operate operate(serviceToken_);
      return schedule_->totalEventsPassed();
    }

    /// Return the number of events that have not passed any trigger.
    /// (N.B. totalEventsFailed() + totalEventsPassed() == totalEvents()
    int totalEventsFailed() const {
      ServiceRegistry::Operate operate(serviceToken_);
      return schedule_->totalEventsFailed();
    }

    /// Turn end_paths "off" if "active" is false;
    /// Turn end_paths "on" if "active" is true.
    void enableEndPaths(bool active) {
      ServiceRegistry::Operate operate(serviceToken_);
      schedule_->enableEndPaths(active);
      for_all(subProcesses_, [active](auto& subProcess) { subProcess.enableEndPaths(active); });
    }

    /// Return true if end_paths are active, and false if they are inactive.
    bool endPathsEnabled() const {
      ServiceRegistry::Operate operate(serviceToken_);
      return schedule_->endPathsEnabled();
    }

    /// Return the trigger report information on paths,
    /// modules-in-path, modules-in-endpath, and modules.
    void getTriggerReport(TriggerReport& rep) const {
      ServiceRegistry::Operate operate(serviceToken_);
      schedule_->getTriggerReport(rep);
    }

    /// Return whether each output module has reached its maximum count.
    /// If there is a subprocess, get this information from the subprocess.
    bool terminate() const {
      ServiceRegistry::Operate operate(serviceToken_);
      if (schedule_->terminate()) {
        return true;
      }
      for (auto const& subProcess : subProcesses_) {
        if (subProcess.terminate()) {
          return true;
        }
      }
      return false;
    }

    ///  Clear all the counters in the trigger report.
    void clearCounters() {
      ServiceRegistry::Operate operate(serviceToken_);
      schedule_->clearCounters();
      for_all(subProcesses_, [](auto& subProcess) { subProcess.clearCounters(); });
    }

  private:
    void beginJob();
    void endJob();
    void processAsync(WaitingTaskHolder iHolder,
                      EventPrincipal const& e,
                      std::vector<std::shared_ptr<const EventSetupImpl>> const*);
    void beginRun(RunPrincipal const& r, IOVSyncValue const& ts);
    void endRun(RunPrincipal const& r, IOVSyncValue const& ts, bool cleaningUpAfterException);
    void beginLuminosityBlock(LuminosityBlockPrincipal const& lb, IOVSyncValue const& ts);
    void endLuminosityBlock(LuminosityBlockPrincipal const& lb, IOVSyncValue const& ts, bool cleaningUpAfterException);

    void propagateProducts(BranchType type, Principal const& parentPrincipal, Principal& principal) const;
    void fixBranchIDListsForEDAliases(
        std::map<BranchID::value_type, BranchID::value_type> const& droppedBranchIDToKeptBranchID);
    void keepThisBranch(BranchDescription const& desc,
                        std::map<BranchID, BranchDescription const*>& trueBranchIDToKeptBranchDesc,
                        std::set<BranchID>& keptProductsInEvent);

    std::map<BranchID::value_type, BranchID::value_type> const& droppedBranchIDToKeptBranchID() {
      return droppedBranchIDToKeptBranchID_;
    }

    std::shared_ptr<BranchIDListHelper const> branchIDListHelper() const {
      return get_underlying_safe(branchIDListHelper_);
    }
    std::shared_ptr<BranchIDListHelper>& branchIDListHelper() { return get_underlying_safe(branchIDListHelper_); }
    std::shared_ptr<ThinnedAssociationsHelper const> thinnedAssociationsHelper() const {
      return get_underlying_safe(thinnedAssociationsHelper_);
    }
    std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper() {
      return get_underlying_safe(thinnedAssociationsHelper_);
    }

    std::shared_ptr<ActivityRegistry> actReg_;  // We do not use propagate_const because the registry itself is mutable.
    ServiceToken serviceToken_;
    std::shared_ptr<ProductRegistry const> parentPreg_;
    std::shared_ptr<ProductRegistry const> preg_;
    edm::propagate_const<std::shared_ptr<BranchIDListHelper>> branchIDListHelper_;
    edm::propagate_const<std::shared_ptr<ThinnedAssociationsHelper>> thinnedAssociationsHelper_;
    edm::propagate_const<std::shared_ptr<SubProcessParentageHelper>> subProcessParentageHelper_;
    std::unique_ptr<ExceptionToActionTable const> act_table_;
    std::shared_ptr<ProcessConfiguration const> processConfiguration_;
    ProcessContext processContext_;
    PathsAndConsumesOfModules pathsAndConsumesOfModules_;
    //We require 1 history for each Run, Lumi and Stream
    // The vectors first hold Stream info, then Lumi then Run
    unsigned int historyLumiOffset_;
    unsigned int historyRunOffset_;
    std::vector<ProcessHistoryRegistry> processHistoryRegistries_;
    std::vector<HistoryAppender> historyAppenders_;
    PrincipalCache principalCache_;
    //vector index is principal lumi's index value
    std::vector<std::shared_ptr<LuminosityBlockPrincipal>> inUseLumiPrincipals_;
    edm::propagate_const<std::shared_ptr<eventsetup::EventSetupProvider>> esp_;
    edm::propagate_const<std::unique_ptr<Schedule>> schedule_;
    std::map<ProcessHistoryID, ProcessHistoryID> parentToChildPhID_;
    std::vector<SubProcess> subProcesses_;
    edm::propagate_const<std::unique_ptr<ParameterSet>> processParameterSet_;

    // keptProducts_ are pointers to the BranchDescription objects describing
    // the branches we are to write.
    //
    // We do not own the BranchDescriptions to which we point.
    SelectedProductsForBranchType keptProducts_;
    ProductSelectorRules productSelectorRules_;
    ProductSelector productSelector_;

    //EventSelection
    bool wantAllEvents_;
    ParameterSetID selector_config_id_;
    mutable detail::TriggerResultsBasedEventSelector selectors_;

    // needed because of possible EDAliases.
    // filled in only if key and value are different.
    std::map<BranchID::value_type, BranchID::value_type> droppedBranchIDToKeptBranchID_;
  };

  // free function
  std::vector<ParameterSet> popSubProcessVParameterSet(ParameterSet& parameterSet);
}  // namespace edm
#endif
