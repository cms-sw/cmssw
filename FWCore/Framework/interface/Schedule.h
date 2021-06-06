#ifndef FWCore_Framework_Schedule_h
#define FWCore_Framework_Schedule_h

/*
  Author: Jim Kowalkowski  28-01-06

  A class for creating a schedule based on paths in the configuration file.
  The schedule is maintained as a sequence of paths.
  After construction, events can be fed to the object and passed through
  all the modules in the schedule.  All accounting about processing
  of events by modules and paths is contained here or in object held
  by containment.

  The trigger results producer and product are generated and managed here.
  This class also manages endpaths and calls to endjob and beginjob.
  Endpaths are just treated as a simple list of modules that need to
  do processing of the event and do not participate in trigger path
  activities.

  This class requires the high-level process pset.  It uses @process_name.
  If the high-level pset contains an "options" pset, then the
  following optional parameter can be present:
  bool wantSummary = true/false   # default false

  wantSummary indicates whether or not the pass/fail/error stats
  for modules and paths should be printed at the end-of-job.

  A TriggerResults object will always be inserted into the event
  for any schedule.  The producer of the TriggerResults EDProduct
  is always the first module in the endpath.  The TriggerResultInserter
  is given a fixed label of "TriggerResults".

  Processing of an event happens by pushing the event through the Paths.
  The scheduler performs the reset() on each of the workers independent
  of the Path objects.

  ------------------------

  About Paths:
  Paths fit into two categories:
  1) trigger paths that contribute directly to saved trigger bits
  2) end paths
  The Schedule holds these paths in two data structures:
  1) main path list
  2) end path list

  Trigger path processing always precedes endpath processing.
  The order of the paths from the input configuration is
  preserved in the main paths list.

  ------------------------

  The Schedule uses the TriggerNamesService to get the names of the
  trigger paths and end paths. When a TriggerResults object is created
  the results are stored in the same order as the trigger names from
  TriggerNamesService.

*/

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/ExceptionHelpers.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/WorkerManager.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/src/GlobalSchedule.h"
#include "FWCore/Framework/src/StreamSchedule.h"
#include "FWCore/Framework/src/SystemTimeKeeper.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <utility>

namespace edm {

  namespace service {
    class TriggerNamesService;
  }
  namespace evetnsetup {
    class ESRecordsToProxyIndices;
  }

  class ActivityRegistry;
  class BranchIDListHelper;
  class EventTransitionInfo;
  class ExceptionCollector;
  class MergeableRunProductMetadata;
  class OutputModuleCommunicator;
  class ProcessContext;
  class ProductRegistry;
  class PreallocationConfiguration;
  class StreamSchedule;
  class GlobalSchedule;
  struct TriggerTimingReport;
  class ModuleRegistry;
  class ThinnedAssociationsHelper;
  class SubProcessParentageHelper;
  class TriggerResultInserter;
  class PathStatusInserter;
  class EndPathStatusInserter;
  class WaitingTaskHolder;

  class Schedule {
  public:
    typedef std::vector<std::string> vstring;
    typedef std::vector<Worker*> AllWorkers;
    typedef std::vector<edm::propagate_const<std::shared_ptr<OutputModuleCommunicator>>> AllOutputModuleCommunicators;

    typedef std::vector<Worker*> Workers;

    Schedule(ParameterSet& proc_pset,
             service::TriggerNamesService const& tns,
             ProductRegistry& pregistry,
             BranchIDListHelper& branchIDListHelper,
             ThinnedAssociationsHelper& thinnedAssociationsHelper,
             SubProcessParentageHelper const* subProcessParentageHelper,
             ExceptionToActionTable const& actions,
             std::shared_ptr<ActivityRegistry> areg,
             std::shared_ptr<ProcessConfiguration> processConfiguration,
             bool hasSubprocesses,
             PreallocationConfiguration const& config,
             ProcessContext const* processContext);

    void processOneEventAsync(WaitingTaskHolder iTask,
                              unsigned int iStreamID,
                              EventTransitionInfo&,
                              ServiceToken const& token);

    template <typename T>
    void processOneGlobalAsync(WaitingTaskHolder iTask,
                               typename T::TransitionInfoType& transitionInfo,
                               ServiceToken const& token,
                               bool cleaningUpAfterException = false);

    template <typename T>
    void processOneStreamAsync(WaitingTaskHolder iTask,
                               unsigned int iStreamID,
                               typename T::TransitionInfoType& transitionInfo,
                               ServiceToken const& token,
                               bool cleaningUpAfterException = false);

    void beginJob(ProductRegistry const&, eventsetup::ESRecordsToProxyIndices const&);
    void endJob(ExceptionCollector& collector);

    void beginStream(unsigned int);
    void endStream(unsigned int);

    // Write the luminosity block
    void writeLumiAsync(WaitingTaskHolder iTask,
                        LuminosityBlockPrincipal const& lbp,
                        ProcessContext const*,
                        ActivityRegistry*);

    // Write the run
    void writeRunAsync(WaitingTaskHolder iTask,
                       RunPrincipal const& rp,
                       ProcessContext const*,
                       ActivityRegistry*,
                       MergeableRunProductMetadata const*);

    void writeProcessBlockAsync(WaitingTaskHolder iTask,
                                ProcessBlockPrincipal const&,
                                ProcessContext const*,
                                ActivityRegistry*);

    // Call closeFile() on all OutputModules.
    void closeOutputFiles();

    // Call openFiles() on all OutputModules
    void openOutputFiles(FileBlock& fb);

    // Call respondToOpenInputFile() on all Modules
    void respondToOpenInputFile(FileBlock const& fb);

    // Call respondToCloseInputFile() on all Modules
    void respondToCloseInputFile(FileBlock const& fb);

    // Call shouldWeCloseFile() on all OutputModules.
    bool shouldWeCloseOutput() const;

    /// Return a vector allowing const access to all the
    /// ModuleDescriptions for this Schedule.

    /// *** N.B. *** Ownership of the ModuleDescriptions is *not*
    /// *** passed to the caller. Do not call delete on these
    /// *** pointers!
    std::vector<ModuleDescription const*> getAllModuleDescriptions() const;

    ///adds to oLabelsToFill the labels for all paths in the process
    void availablePaths(std::vector<std::string>& oLabelsToFill) const;

    ///Adds to oLabelsToFill the labels for all trigger paths in the process.
    ///This is different from availablePaths because it includes the
    ///empty paths to match the entries in TriggerResults exactly.
    void triggerPaths(std::vector<std::string>& oLabelsToFill) const;

    ///adds to oLabelsToFill the labels for all end paths in the process
    void endPaths(std::vector<std::string>& oLabelsToFill) const;

    ///adds to oLabelsToFill in execution order the labels of all modules in path iPathLabel
    void modulesInPath(std::string const& iPathLabel, std::vector<std::string>& oLabelsToFill) const;

    ///adds the ModuleDescriptions into the vector for the modules scheduled in path iPathLabel
    ///hint is a performance optimization if you might know the position of the module in the path
    void moduleDescriptionsInPath(std::string const& iPathLabel,
                                  std::vector<ModuleDescription const*>& descriptions,
                                  unsigned int hint) const;

    ///adds the ModuleDescriptions into the vector for the modules scheduled in path iEndPathLabel
    ///hint is a performance optimization if you might know the position of the module in the path
    void moduleDescriptionsInEndPath(std::string const& iEndPathLabel,
                                     std::vector<ModuleDescription const*>& descriptions,
                                     unsigned int hint) const;

    void fillModuleAndConsumesInfo(
        std::vector<ModuleDescription const*>& allModuleDescriptions,
        std::vector<std::pair<unsigned int, unsigned int>>& moduleIDToIndex,
        std::array<std::vector<std::vector<ModuleDescription const*>>, NumBranchTypes>&
            modulesWhoseProductsAreConsumedBy,
        std::vector<std::vector<ModuleProcessName>>& modulesInPreviousProcessesWhoseProductsAreConsumedBy,
        ProductRegistry const& preg) const;

    /// Return the number of events this Schedule has tried to process
    /// (inclues both successes and failures, including failures due
    /// to exceptions during processing).
    int totalEvents() const;

    /// Return the number of events which have been passed by one or
    /// more trigger paths.
    int totalEventsPassed() const;

    /// Return the number of events that have not passed any trigger.
    /// (N.B. totalEventsFailed() + totalEventsPassed() == totalEvents()
    int totalEventsFailed() const;

    /// Turn end_paths "off" if "active" is false;
    /// turn end_paths "on" if "active" is true.
    void enableEndPaths(bool active);

    /// Return true if end_paths are active, and false if they are
    /// inactive.
    bool endPathsEnabled() const;

    /// Return the trigger report information on paths,
    /// modules-in-path, modules-in-endpath, and modules.
    void getTriggerReport(TriggerReport& rep) const;

    /// Return the trigger timing report information on paths,
    /// modules-in-path, modules-in-endpath, and modules.
    void getTriggerTimingReport(TriggerTimingReport& rep) const;

    /// Return whether each output module has reached its maximum count.
    bool terminate() const;

    ///  Clear all the counters in the trigger report.
    void clearCounters();

    /// clone the type of module with label iLabel but configure with iPSet.
    /// Returns true if successful.
    bool changeModule(std::string const& iLabel,
                      ParameterSet const& iPSet,
                      const ProductRegistry& iRegistry,
                      eventsetup::ESRecordsToProxyIndices const&);

    /// Deletes module with label iLabel
    void deleteModule(std::string const& iLabel, ActivityRegistry* areg);

    /// returns the collection of pointers to workers
    AllWorkers const& allWorkers() const;

    /// Convert "@currentProcess" in InputTag process names to the actual current process name.
    void convertCurrentProcessAlias(std::string const& processName);

  private:
    void limitOutput(ParameterSet const& proc_pset,
                     BranchIDLists const& branchIDLists,
                     SubProcessParentageHelper const* subProcessParentageHelper);

    std::shared_ptr<TriggerResultInserter const> resultsInserter() const {
      return get_underlying_safe(resultsInserter_);
    }
    std::shared_ptr<TriggerResultInserter>& resultsInserter() { return get_underlying_safe(resultsInserter_); }
    std::shared_ptr<ModuleRegistry const> moduleRegistry() const { return get_underlying_safe(moduleRegistry_); }
    std::shared_ptr<ModuleRegistry>& moduleRegistry() { return get_underlying_safe(moduleRegistry_); }

    edm::propagate_const<std::shared_ptr<TriggerResultInserter>> resultsInserter_;
    std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>> pathStatusInserters_;
    std::vector<edm::propagate_const<std::shared_ptr<EndPathStatusInserter>>> endPathStatusInserters_;
    edm::propagate_const<std::shared_ptr<ModuleRegistry>> moduleRegistry_;
    std::vector<edm::propagate_const<std::shared_ptr<StreamSchedule>>> streamSchedules_;
    //In the future, we will have one GlobalSchedule per simultaneous transition
    edm::propagate_const<std::unique_ptr<GlobalSchedule>> globalSchedule_;

    AllOutputModuleCommunicators all_output_communicators_;
    PreallocationConfiguration preallocConfig_;

    edm::propagate_const<std::unique_ptr<SystemTimeKeeper>> summaryTimeKeeper_;

    std::vector<std::string> const* pathNames_;
    std::vector<std::string> const* endPathNames_;
    bool wantSummary_;

    volatile bool endpathsAreActive_;
  };

  template <typename T>
  void Schedule::processOneStreamAsync(WaitingTaskHolder iTaskHolder,
                                       unsigned int iStreamID,
                                       typename T::TransitionInfoType& transitionInfo,
                                       ServiceToken const& token,
                                       bool cleaningUpAfterException) {
    assert(iStreamID < streamSchedules_.size());
    streamSchedules_[iStreamID]->processOneStreamAsync<T>(
        std::move(iTaskHolder), transitionInfo, token, cleaningUpAfterException);
  }

  template <typename T>
  void Schedule::processOneGlobalAsync(WaitingTaskHolder iTaskHolder,
                                       typename T::TransitionInfoType& transitionInfo,
                                       ServiceToken const& token,
                                       bool cleaningUpAfterException) {
    globalSchedule_->processOneGlobalAsync<T>(iTaskHolder, transitionInfo, token, cleaningUpAfterException);
  }

}  // namespace edm
#endif
