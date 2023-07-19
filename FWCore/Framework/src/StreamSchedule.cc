#include "FWCore/Framework/interface/StreamSchedule.h"

#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "FWCore/Framework/src/OutputModuleDescription.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/src/TriggerReport.h"
#include "FWCore/Framework/src/TriggerTimingReport.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/interface/OutputModuleCommunicator.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/Framework/src/PathStatusInserter.h"
#include "FWCore/Framework/src/EndPathStatusInserter.h"
#include "FWCore/Framework/interface/WorkerInPath.h"
#include "FWCore/Framework/interface/maker/ModuleHolder.h"
#include "FWCore/Framework/interface/maker/WorkerT.h"
#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

#include "LuminosityBlockProcessingStatus.h"
#include "processEDAliases.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <exception>
#include <fmt/format.h>

namespace edm {

  namespace {

    // Function template to transform each element in the input range to
    // a value placed into the output range. The supplied function
    // should take a const_reference to the 'input', and write to a
    // reference to the 'output'.
    template <typename InputIterator, typename ForwardIterator, typename Func>
    void transform_into(InputIterator begin, InputIterator end, ForwardIterator out, Func func) {
      for (; begin != end; ++begin, ++out)
        func(*begin, *out);
    }

    // Function template that takes a sequence 'from', a sequence
    // 'to', and a callable object 'func'. It and applies
    // transform_into to fill the 'to' sequence with the values
    // calcuated by the callable object, taking care to fill the
    // outupt only if all calls succeed.
    template <typename FROM, typename TO, typename FUNC>
    void fill_summary(FROM const& from, TO& to, FUNC func) {
      if (to.size() != from.size()) {
        TO temp(from.size());
        transform_into(from.begin(), from.end(), temp.begin(), func);
        to.swap(temp);
      } else {
        transform_into(from.begin(), from.end(), to.begin(), func);
      }
    }

    // -----------------------------

    // Here we make the trigger results inserter directly.  This should
    // probably be a utility in the WorkerRegistry or elsewhere.

    StreamSchedule::WorkerPtr makeInserter(ExceptionToActionTable const& actions,
                                           std::shared_ptr<ActivityRegistry> areg,
                                           std::shared_ptr<TriggerResultInserter> inserter) {
      StreamSchedule::WorkerPtr ptr(
          new edm::WorkerT<TriggerResultInserter::ModuleType>(inserter, inserter->moduleDescription(), &actions));
      ptr->setActivityRegistry(areg);
      return ptr;
    }

    void initializeBranchToReadingWorker(std::vector<std::string> const& branchesToDeleteEarly,
                                         ProductRegistry const& preg,
                                         std::multimap<std::string, Worker*>& branchToReadingWorker) {
      auto vBranchesToDeleteEarly = branchesToDeleteEarly;
      // Remove any duplicates
      std::sort(vBranchesToDeleteEarly.begin(), vBranchesToDeleteEarly.end(), std::less<std::string>());
      vBranchesToDeleteEarly.erase(std::unique(vBranchesToDeleteEarly.begin(), vBranchesToDeleteEarly.end()),
                                   vBranchesToDeleteEarly.end());

      // Are the requested items in the product registry?
      auto allBranchNames = preg.allBranchNames();
      //the branch names all end with a period, which we do not want to compare with
      for (auto& b : allBranchNames) {
        b.resize(b.size() - 1);
      }
      std::sort(allBranchNames.begin(), allBranchNames.end(), std::less<std::string>());
      std::vector<std::string> temp;
      temp.reserve(vBranchesToDeleteEarly.size());

      std::set_intersection(vBranchesToDeleteEarly.begin(),
                            vBranchesToDeleteEarly.end(),
                            allBranchNames.begin(),
                            allBranchNames.end(),
                            std::back_inserter(temp));
      vBranchesToDeleteEarly.swap(temp);
      if (temp.size() != vBranchesToDeleteEarly.size()) {
        std::vector<std::string> missingProducts;
        std::set_difference(temp.begin(),
                            temp.end(),
                            vBranchesToDeleteEarly.begin(),
                            vBranchesToDeleteEarly.end(),
                            std::back_inserter(missingProducts));
        LogInfo l("MissingProductsForCanDeleteEarly");
        l << "The following products in the 'canDeleteEarly' list are not available in this job and will be ignored.";
        for (auto const& n : missingProducts) {
          l << "\n " << n;
        }
      }
      //set placeholder for the branch, we will remove the nullptr if a
      // module actually wants the branch.
      for (auto const& branch : vBranchesToDeleteEarly) {
        branchToReadingWorker.insert(std::make_pair(branch, static_cast<Worker*>(nullptr)));
      }
    }

    Worker* getWorker(std::string const& moduleLabel,
                      ParameterSet& proc_pset,
                      WorkerManager& workerManager,
                      ProductRegistry& preg,
                      PreallocationConfiguration const* prealloc,
                      std::shared_ptr<ProcessConfiguration const> processConfiguration) {
      bool isTracked;
      ParameterSet* modpset = proc_pset.getPSetForUpdate(moduleLabel, isTracked);
      if (modpset == nullptr) {
        return nullptr;
      }
      assert(isTracked);

      return workerManager.getWorker(*modpset, preg, prealloc, processConfiguration, moduleLabel);
    }

    // If ConditionalTask modules exist in the container of module
    // names, returns the range (std::pair) for the modules. The range
    // excludes the special markers '#' (right before the
    // ConditionalTask modules) and '@' (last element).
    // If the module name container does not contain ConditionalTask
    // modules, returns std::pair of end iterators.
    template <typename T>
    auto findConditionalTaskModulesRange(T& modnames) {
      auto beg = std::find(modnames.begin(), modnames.end(), "#");
      if (beg == modnames.end()) {
        return std::pair(modnames.end(), modnames.end());
      }
      return std::pair(beg + 1, std::prev(modnames.end()));
    }

    std::optional<std::string> findBestMatchingAlias(
        std::unordered_multimap<std::string, edm::BranchDescription const*> const& conditionalModuleBranches,
        std::unordered_multimap<std::string, StreamSchedule::AliasInfo> const& aliasMap,
        std::string const& productModuleLabel,
        ConsumesInfo const& consumesInfo) {
      std::optional<std::string> best;
      int wildcardsInBest = std::numeric_limits<int>::max();
      bool bestIsAmbiguous = false;

      auto updateBest = [&best, &wildcardsInBest, &bestIsAmbiguous](
                            std::string const& label, bool instanceIsWildcard, bool typeIsWildcard) {
        int const wildcards = static_cast<int>(instanceIsWildcard) + static_cast<int>(typeIsWildcard);
        if (wildcards == 0) {
          bestIsAmbiguous = false;
          return true;
        }
        if (not best or wildcards < wildcardsInBest) {
          best = label;
          wildcardsInBest = wildcards;
          bestIsAmbiguous = false;
        } else if (best and *best != label and wildcardsInBest == wildcards) {
          bestIsAmbiguous = true;
        }
        return false;
      };

      auto findAlias = aliasMap.equal_range(productModuleLabel);
      for (auto it = findAlias.first; it != findAlias.second; ++it) {
        std::string const& aliasInstanceLabel =
            it->second.instanceLabel != "*" ? it->second.instanceLabel : it->second.originalInstanceLabel;
        bool const instanceIsWildcard = (aliasInstanceLabel == "*");
        if (instanceIsWildcard or consumesInfo.instance() == aliasInstanceLabel) {
          bool const typeIsWildcard = it->second.friendlyClassName == "*";
          if (typeIsWildcard or (consumesInfo.type().friendlyClassName() == it->second.friendlyClassName)) {
            if (updateBest(it->second.originalModuleLabel, instanceIsWildcard, typeIsWildcard)) {
              return it->second.originalModuleLabel;
            }
          } else if (consumesInfo.kindOfType() == ELEMENT_TYPE) {
            //consume is a View so need to do more intrusive search
            //find matching branches in module
            auto branches = conditionalModuleBranches.equal_range(productModuleLabel);
            for (auto itBranch = branches.first; itBranch != branches.second; ++it) {
              if (typeIsWildcard or itBranch->second->productInstanceName() == it->second.originalInstanceLabel) {
                if (productholderindexhelper::typeIsViewCompatible(consumesInfo.type(),
                                                                   TypeID(itBranch->second->wrappedType().typeInfo()),
                                                                   itBranch->second->className())) {
                  if (updateBest(it->second.originalModuleLabel, instanceIsWildcard, typeIsWildcard)) {
                    return it->second.originalModuleLabel;
                  }
                }
              }
            }
          }
        }
      }
      if (bestIsAmbiguous) {
        throw Exception(errors::UnimplementedFeature)
            << "Encountered ambiguity when trying to find a best-matching alias for\n"
            << " friendly class name " << consumesInfo.type().friendlyClassName() << "\n"
            << " module label " << productModuleLabel << "\n"
            << " product instance name " << consumesInfo.instance() << "\n"
            << "when processing EDAliases for modules in ConditionalTasks. Two aliases have the same number of "
               "wildcards ("
            << wildcardsInBest << ")";
      }
      return best;
    }
  }  // namespace

  // -----------------------------

  typedef std::vector<std::string> vstring;

  // -----------------------------

  class ConditionalTaskHelper {
  public:
    using AliasInfo = StreamSchedule::AliasInfo;

    ConditionalTaskHelper(ParameterSet& proc_pset,
                          ProductRegistry& preg,
                          PreallocationConfiguration const* prealloc,
                          std::shared_ptr<ProcessConfiguration const> processConfiguration,
                          WorkerManager& workerManager,
                          std::vector<std::string> const& trigPathNames) {
      std::unordered_set<std::string> allConditionalMods;
      for (auto const& pathName : trigPathNames) {
        auto const modnames = proc_pset.getParameter<vstring>(pathName);

        //Pull out ConditionalTask modules
        auto condRange = findConditionalTaskModulesRange(modnames);
        if (condRange.first == condRange.second)
          continue;

        //the last entry should be ignored since it is required to be "@"
        allConditionalMods.insert(condRange.first, condRange.second);
      }

      for (auto const& cond : allConditionalMods) {
        //force the creation of the conditional modules so alias check can work
        (void)getWorker(cond, proc_pset, workerManager, preg, prealloc, processConfiguration);
      }

      fillAliasMap(proc_pset, allConditionalMods);
      processSwitchEDAliases(proc_pset, preg, *processConfiguration, allConditionalMods);

      //find branches created by the conditional modules
      for (auto const& prod : preg.productList()) {
        if (allConditionalMods.find(prod.first.moduleLabel()) != allConditionalMods.end()) {
          conditionalModsBranches_.emplace(prod.first.moduleLabel(), &prod.second);
        }
      }
    }

    std::unordered_multimap<std::string, AliasInfo> const& aliasMap() const { return aliasMap_; }

    std::unordered_multimap<std::string, edm::BranchDescription const*> conditionalModuleBranches(
        std::unordered_set<std::string> const& conditionalmods) const {
      std::unordered_multimap<std::string, edm::BranchDescription const*> ret;
      for (auto const& mod : conditionalmods) {
        auto range = conditionalModsBranches_.equal_range(mod);
        ret.insert(range.first, range.second);
      }
      return ret;
    }

  private:
    void fillAliasMap(ParameterSet const& proc_pset, std::unordered_set<std::string> const& allConditionalMods) {
      auto aliases = proc_pset.getParameter<std::vector<std::string>>("@all_aliases");
      std::string const star("*");
      for (auto const& alias : aliases) {
        auto info = proc_pset.getParameter<edm::ParameterSet>(alias);
        auto aliasedToModuleLabels = info.getParameterNames();
        for (auto const& mod : aliasedToModuleLabels) {
          if (not mod.empty() and mod[0] != '@' and allConditionalMods.find(mod) != allConditionalMods.end()) {
            auto aliasVPSet = info.getParameter<std::vector<edm::ParameterSet>>(mod);
            for (auto const& aliasPSet : aliasVPSet) {
              std::string type = star;
              std::string instance = star;
              std::string originalInstance = star;
              if (aliasPSet.exists("type")) {
                type = aliasPSet.getParameter<std::string>("type");
              }
              if (aliasPSet.exists("toProductInstance")) {
                instance = aliasPSet.getParameter<std::string>("toProductInstance");
              }
              if (aliasPSet.exists("fromProductInstance")) {
                originalInstance = aliasPSet.getParameter<std::string>("fromProductInstance");
              }

              aliasMap_.emplace(alias, AliasInfo{type, instance, originalInstance, mod});
            }
          }
        }
      }
    }

    void processSwitchEDAliases(ParameterSet const& proc_pset,
                                ProductRegistry& preg,
                                ProcessConfiguration const& processConfiguration,
                                std::unordered_set<std::string> const& allConditionalMods) {
      auto const& all_modules = proc_pset.getParameter<std::vector<std::string>>("@all_modules");
      std::vector<std::string> switchEDAliases;
      for (auto const& module : all_modules) {
        auto const& mod_pset = proc_pset.getParameter<edm::ParameterSet>(module);
        if (mod_pset.getParameter<std::string>("@module_type") == "SwitchProducer") {
          auto const& all_cases = mod_pset.getParameter<std::vector<std::string>>("@all_cases");
          for (auto const& case_label : all_cases) {
            auto range = aliasMap_.equal_range(case_label);
            if (range.first != range.second) {
              switchEDAliases.push_back(case_label);
            }
          }
        }
      }
      detail::processEDAliases(
          switchEDAliases, allConditionalMods, proc_pset, processConfiguration.processName(), preg);
    }

    std::unordered_multimap<std::string, AliasInfo> aliasMap_;
    std::unordered_multimap<std::string, edm::BranchDescription const*> conditionalModsBranches_;
  };

  // -----------------------------

  StreamSchedule::StreamSchedule(
      std::shared_ptr<TriggerResultInserter> inserter,
      std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters,
      std::vector<edm::propagate_const<std::shared_ptr<EndPathStatusInserter>>>& endPathStatusInserters,
      std::shared_ptr<ModuleRegistry> modReg,
      ParameterSet& proc_pset,
      service::TriggerNamesService const& tns,
      PreallocationConfiguration const& prealloc,
      ProductRegistry& preg,
      ExceptionToActionTable const& actions,
      std::shared_ptr<ActivityRegistry> areg,
      std::shared_ptr<ProcessConfiguration const> processConfiguration,
      StreamID streamID,
      ProcessContext const* processContext)
      : workerManager_(modReg, areg, actions),
        actReg_(areg),
        results_(new HLTGlobalStatus(tns.getTrigPaths().size())),
        results_inserter_(),
        trig_paths_(),
        end_paths_(),
        total_events_(),
        total_passed_(),
        number_of_unscheduled_modules_(0),
        streamID_(streamID),
        streamContext_(streamID_, processContext),
        skippingEvent_(false) {
    bool hasPath = false;
    std::vector<std::string> const& pathNames = tns.getTrigPaths();
    std::vector<std::string> const& endPathNames = tns.getEndPaths();

    ConditionalTaskHelper conditionalTaskHelper(
        proc_pset, preg, &prealloc, processConfiguration, workerManager_, pathNames);

    int trig_bitpos = 0;
    trig_paths_.reserve(pathNames.size());
    for (auto const& trig_name : pathNames) {
      fillTrigPath(proc_pset,
                   preg,
                   &prealloc,
                   processConfiguration,
                   trig_bitpos,
                   trig_name,
                   results(),
                   endPathNames,
                   conditionalTaskHelper);
      ++trig_bitpos;
      hasPath = true;
    }

    if (hasPath) {
      // the results inserter stands alone
      inserter->setTrigResultForStream(streamID.value(), results());

      results_inserter_ = makeInserter(actions, actReg_, inserter);
      addToAllWorkers(results_inserter_.get());
    }

    // fill normal endpaths
    int bitpos = 0;
    end_paths_.reserve(endPathNames.size());
    for (auto const& end_path_name : endPathNames) {
      fillEndPath(
          proc_pset, preg, &prealloc, processConfiguration, bitpos, end_path_name, endPathNames, conditionalTaskHelper);
      ++bitpos;
    }

    makePathStatusInserters(pathStatusInserters, endPathStatusInserters, actions);

    //See if all modules were used
    std::set<std::string> usedWorkerLabels;
    for (auto const& worker : allWorkers()) {
      usedWorkerLabels.insert(worker->description()->moduleLabel());
    }
    std::vector<std::string> modulesInConfig(proc_pset.getParameter<std::vector<std::string>>("@all_modules"));
    std::set<std::string> modulesInConfigSet(modulesInConfig.begin(), modulesInConfig.end());
    std::vector<std::string> unusedLabels;
    set_difference(modulesInConfigSet.begin(),
                   modulesInConfigSet.end(),
                   usedWorkerLabels.begin(),
                   usedWorkerLabels.end(),
                   back_inserter(unusedLabels));
    std::set<std::string> unscheduledLabels;
    std::vector<std::string> shouldBeUsedLabels;
    if (!unusedLabels.empty()) {
      //Need to
      // 1) create worker
      // 2) if it is a WorkerT<EDProducer>, add it to our list
      // 3) hand list to our delayed reader
      for (auto const& label : unusedLabels) {
        bool isTracked;
        ParameterSet* modulePSet(proc_pset.getPSetForUpdate(label, isTracked));
        assert(isTracked);
        assert(modulePSet != nullptr);
        workerManager_.addToUnscheduledWorkers(
            *modulePSet, preg, &prealloc, processConfiguration, label, unscheduledLabels, shouldBeUsedLabels);
      }
      if (!shouldBeUsedLabels.empty()) {
        std::ostringstream unusedStream;
        unusedStream << "'" << shouldBeUsedLabels.front() << "'";
        for (std::vector<std::string>::iterator itLabel = shouldBeUsedLabels.begin() + 1,
                                                itLabelEnd = shouldBeUsedLabels.end();
             itLabel != itLabelEnd;
             ++itLabel) {
          unusedStream << ",'" << *itLabel << "'";
        }
        LogInfo("path") << "The following module labels are not assigned to any path:\n" << unusedStream.str() << "\n";
      }
    }
    number_of_unscheduled_modules_ = unscheduledLabels.size();
  }  // StreamSchedule::StreamSchedule

  void StreamSchedule::initializeEarlyDelete(ModuleRegistry& modReg,
                                             std::vector<std::string> const& branchesToDeleteEarly,
                                             std::multimap<std::string, std::string> const& referencesToBranches,
                                             std::vector<std::string> const& modulesToSkip,
                                             edm::ProductRegistry const& preg) {
    // setup the list with those products actually registered for this job
    std::multimap<std::string, Worker*> branchToReadingWorker;
    initializeBranchToReadingWorker(branchesToDeleteEarly, preg, branchToReadingWorker);

    const std::vector<std::string> kEmpty;
    std::map<Worker*, unsigned int> reserveSizeForWorker;
    unsigned int upperLimitOnReadingWorker = 0;
    unsigned int upperLimitOnIndicies = 0;
    unsigned int nUniqueBranchesToDelete = branchToReadingWorker.size();

    //talk with output modules first
    modReg.forAllModuleHolders([&branchToReadingWorker, &nUniqueBranchesToDelete](maker::ModuleHolder* iHolder) {
      auto comm = iHolder->createOutputModuleCommunicator();
      if (comm) {
        if (!branchToReadingWorker.empty()) {
          //If an OutputModule needs a product, we can't delete it early
          // so we should remove it from our list
          SelectedProductsForBranchType const& kept = comm->keptProducts();
          for (auto const& item : kept[InEvent]) {
            BranchDescription const& desc = *item.first;
            auto found = branchToReadingWorker.equal_range(desc.branchName());
            if (found.first != found.second) {
              --nUniqueBranchesToDelete;
              branchToReadingWorker.erase(found.first, found.second);
            }
          }
        }
      }
    });

    if (branchToReadingWorker.empty()) {
      return;
    }

    std::unordered_set<std::string> modulesToExclude(modulesToSkip.begin(), modulesToSkip.end());
    for (auto w : allWorkers()) {
      if (modulesToExclude.end() != modulesToExclude.find(w->description()->moduleLabel())) {
        continue;
      }
      //determine if this module could read a branch we want to delete early
      auto consumes = w->consumesInfo();
      if (not consumes.empty()) {
        bool foundAtLeastOneMatchingBranch = false;
        for (auto const& product : consumes) {
          std::string branch = fmt::format("{}_{}_{}_{}",
                                           product.type().friendlyClassName(),
                                           product.label().data(),
                                           product.instance().data(),
                                           product.process().data());
          {
            //Handle case where worker directly consumes product
            auto found = branchToReadingWorker.end();
            if (product.process().empty()) {
              auto startFound = branchToReadingWorker.lower_bound(branch);
              if (startFound != branchToReadingWorker.end()) {
                if (startFound->first.substr(0, branch.size()) == branch) {
                  //match all processNames here, even if it means multiple matches will happen
                  found = startFound;
                }
              }
            } else {
              auto exactFound = branchToReadingWorker.equal_range(branch);
              if (exactFound.first != exactFound.second) {
                found = exactFound.first;
              }
            }
            if (found != branchToReadingWorker.end()) {
              if (not foundAtLeastOneMatchingBranch) {
                ++upperLimitOnReadingWorker;
                foundAtLeastOneMatchingBranch = true;
              }
              ++upperLimitOnIndicies;
              ++reserveSizeForWorker[w];
              if (nullptr == found->second) {
                found->second = w;
              } else {
                branchToReadingWorker.insert(make_pair(found->first, w));
              }
            }
          }
          {
            //Handle case where indirectly consumes product
            auto found = referencesToBranches.end();
            if (product.process().empty()) {
              auto startFound = referencesToBranches.lower_bound(branch);
              if (startFound != referencesToBranches.end()) {
                if (startFound->first.substr(0, branch.size()) == branch) {
                  //match all processNames here, even if it means multiple matches will happen
                  found = startFound;
                }
              }
            } else {
              //can match exactly
              auto exactFound = referencesToBranches.equal_range(branch);
              if (exactFound.first != exactFound.second) {
                found = exactFound.first;
              }
            }
            if (found != referencesToBranches.end()) {
              for (auto itr = found; (itr != referencesToBranches.end()) and (itr->first == found->first); ++itr) {
                auto foundInBranchToReadingWorker = branchToReadingWorker.find(itr->second);
                if (foundInBranchToReadingWorker == branchToReadingWorker.end()) {
                  continue;
                }
                if (not foundAtLeastOneMatchingBranch) {
                  ++upperLimitOnReadingWorker;
                  foundAtLeastOneMatchingBranch = true;
                }
                ++upperLimitOnIndicies;
                ++reserveSizeForWorker[w];
                if (nullptr == foundInBranchToReadingWorker->second) {
                  foundInBranchToReadingWorker->second = w;
                } else {
                  branchToReadingWorker.insert(make_pair(itr->second, w));
                }
              }
            }
          }
        }
      }
    }
    {
      auto it = branchToReadingWorker.begin();
      std::vector<std::string> unusedBranches;
      while (it != branchToReadingWorker.end()) {
        if (it->second == nullptr) {
          unusedBranches.push_back(it->first);
          //erasing the object invalidates the iterator so must advance it first
          auto temp = it;
          ++it;
          branchToReadingWorker.erase(temp);
        } else {
          ++it;
        }
      }
      if (not unusedBranches.empty()) {
        LogWarning l("UnusedProductsForCanDeleteEarly");
        l << "The following products in the 'canDeleteEarly' list are not used in this job and will be ignored.\n"
             " If possible, remove the producer from the job.";
        for (auto const& n : unusedBranches) {
          l << "\n " << n;
        }
      }
    }
    if (!branchToReadingWorker.empty()) {
      earlyDeleteHelpers_.reserve(upperLimitOnReadingWorker);
      earlyDeleteHelperToBranchIndicies_.resize(upperLimitOnIndicies, 0);
      earlyDeleteBranchToCount_.reserve(nUniqueBranchesToDelete);
      std::map<const Worker*, EarlyDeleteHelper*> alreadySeenWorkers;
      std::string lastBranchName;
      size_t nextOpenIndex = 0;
      unsigned int* beginAddress = &(earlyDeleteHelperToBranchIndicies_.front());
      for (auto& branchAndWorker : branchToReadingWorker) {
        if (lastBranchName != branchAndWorker.first) {
          //have to put back the period we removed earlier in order to get the proper name
          BranchID bid(branchAndWorker.first + ".");
          earlyDeleteBranchToCount_.emplace_back(bid, 0U);
          lastBranchName = branchAndWorker.first;
        }
        auto found = alreadySeenWorkers.find(branchAndWorker.second);
        if (alreadySeenWorkers.end() == found) {
          //NOTE: we will set aside enough space in earlyDeleteHelperToBranchIndicies_ to accommodate
          // all the branches that might be read by this worker. However, initially we will only tell the
          // EarlyDeleteHelper about the first one. As additional branches are added via 'appendIndex' the
          // EarlyDeleteHelper will automatically advance its internal end pointer.
          size_t index = nextOpenIndex;
          size_t nIndices = reserveSizeForWorker[branchAndWorker.second];
          assert(index < earlyDeleteHelperToBranchIndicies_.size());
          earlyDeleteHelperToBranchIndicies_[index] = earlyDeleteBranchToCount_.size() - 1;
          earlyDeleteHelpers_.emplace_back(beginAddress + index, beginAddress + index + 1, &earlyDeleteBranchToCount_);
          branchAndWorker.second->setEarlyDeleteHelper(&(earlyDeleteHelpers_.back()));
          alreadySeenWorkers.insert(std::make_pair(branchAndWorker.second, &(earlyDeleteHelpers_.back())));
          nextOpenIndex += nIndices;
        } else {
          found->second->appendIndex(earlyDeleteBranchToCount_.size() - 1);
        }
      }

      //Now we can compactify the earlyDeleteHelperToBranchIndicies_ since we may have over estimated the
      // space needed for each module
      auto itLast = earlyDeleteHelpers_.begin();
      for (auto it = earlyDeleteHelpers_.begin() + 1; it != earlyDeleteHelpers_.end(); ++it) {
        if (itLast->end() != it->begin()) {
          //figure the offset for next Worker since it hasn't been moved yet so it has the original address
          unsigned int delta = it->begin() - itLast->end();
          it->shiftIndexPointers(delta);

          earlyDeleteHelperToBranchIndicies_.erase(
              earlyDeleteHelperToBranchIndicies_.begin() + (itLast->end() - beginAddress),
              earlyDeleteHelperToBranchIndicies_.begin() + (it->begin() - beginAddress));
        }
        itLast = it;
      }
      earlyDeleteHelperToBranchIndicies_.erase(
          earlyDeleteHelperToBranchIndicies_.begin() + (itLast->end() - beginAddress),
          earlyDeleteHelperToBranchIndicies_.end());

      //now tell the paths about the deleters
      for (auto& p : trig_paths_) {
        p.setEarlyDeleteHelpers(alreadySeenWorkers);
      }
      for (auto& p : end_paths_) {
        p.setEarlyDeleteHelpers(alreadySeenWorkers);
      }
      resetEarlyDelete();
    }
  }

  std::vector<Worker*> StreamSchedule::tryToPlaceConditionalModules(
      Worker* worker,
      std::unordered_set<std::string>& conditionalModules,
      std::unordered_multimap<std::string, edm::BranchDescription const*> const& conditionalModuleBranches,
      std::unordered_multimap<std::string, AliasInfo> const& aliasMap,
      ParameterSet& proc_pset,
      ProductRegistry& preg,
      PreallocationConfiguration const* prealloc,
      std::shared_ptr<ProcessConfiguration const> processConfiguration) {
    std::vector<Worker*> returnValue;
    auto const& consumesInfo = worker->consumesInfo();
    auto moduleLabel = worker->description()->moduleLabel();
    using namespace productholderindexhelper;
    for (auto const& ci : consumesInfo) {
      if (not ci.skipCurrentProcess() and
          (ci.process().empty() or ci.process() == processConfiguration->processName())) {
        auto productModuleLabel = std::string(ci.label());
        bool productFromConditionalModule = false;
        auto itFound = conditionalModules.find(productModuleLabel);
        if (itFound == conditionalModules.end()) {
          //Check to see if this was an alias
          //note that aliasMap was previously filtered so only the conditional modules remain there
          auto foundAlias = findBestMatchingAlias(conditionalModuleBranches, aliasMap, productModuleLabel, ci);
          if (foundAlias) {
            productModuleLabel = *foundAlias;
            productFromConditionalModule = true;
            itFound = conditionalModules.find(productModuleLabel);
            //check that the alias-for conditional module has not been used
            if (itFound == conditionalModules.end()) {
              continue;
            }
          }
        } else {
          //need to check the rest of the data product info
          auto findBranches = conditionalModuleBranches.equal_range(productModuleLabel);
          for (auto itBranch = findBranches.first; itBranch != findBranches.second; ++itBranch) {
            if (itBranch->second->productInstanceName() == ci.instance()) {
              if (ci.kindOfType() == PRODUCT_TYPE) {
                if (ci.type() == itBranch->second->unwrappedTypeID()) {
                  productFromConditionalModule = true;
                  break;
                }
              } else {
                //this is a view
                if (typeIsViewCompatible(
                        ci.type(), TypeID(itBranch->second->wrappedType().typeInfo()), itBranch->second->className())) {
                  productFromConditionalModule = true;
                  break;
                }
              }
            }
          }
        }
        if (productFromConditionalModule) {
          auto condWorker =
              getWorker(productModuleLabel, proc_pset, workerManager_, preg, prealloc, processConfiguration);
          assert(condWorker);

          conditionalModules.erase(itFound);

          auto dependents = tryToPlaceConditionalModules(condWorker,
                                                         conditionalModules,
                                                         conditionalModuleBranches,
                                                         aliasMap,
                                                         proc_pset,
                                                         preg,
                                                         prealloc,
                                                         processConfiguration);
          returnValue.insert(returnValue.end(), dependents.begin(), dependents.end());
          returnValue.push_back(condWorker);
        }
      }
    }
    return returnValue;
  }

  void StreamSchedule::fillWorkers(ParameterSet& proc_pset,
                                   ProductRegistry& preg,
                                   PreallocationConfiguration const* prealloc,
                                   std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                   std::string const& pathName,
                                   bool ignoreFilters,
                                   PathWorkers& out,
                                   std::vector<std::string> const& endPathNames,
                                   ConditionalTaskHelper const& conditionalTaskHelper) {
    vstring modnames = proc_pset.getParameter<vstring>(pathName);
    PathWorkers tmpworkers;

    //Pull out ConditionalTask modules
    auto condRange = findConditionalTaskModulesRange(modnames);

    std::unordered_set<std::string> conditionalmods;
    //An EDAlias may be redirecting to a module on a ConditionalTask
    std::unordered_multimap<std::string, edm::BranchDescription const*> conditionalModsBranches;
    std::unordered_map<std::string, unsigned int> conditionalModOrder;
    if (condRange.first != condRange.second) {
      for (auto it = condRange.first; it != condRange.second; ++it) {
        // ordering needs to skip the # token in the path list
        conditionalModOrder.emplace(*it, it - modnames.begin() - 1);
      }
      //the last entry should be ignored since it is required to be "@"
      conditionalmods = std::unordered_set<std::string>(std::make_move_iterator(condRange.first),
                                                        std::make_move_iterator(condRange.second));

      conditionalModsBranches = conditionalTaskHelper.conditionalModuleBranches(conditionalmods);
      modnames.erase(std::prev(condRange.first), modnames.end());
    }

    unsigned int placeInPath = 0;
    for (auto const& name : modnames) {
      //Modules except EDFilters are set to run concurrently by default
      bool doNotRunConcurrently = false;
      WorkerInPath::FilterAction filterAction = WorkerInPath::Normal;
      if (name[0] == '!') {
        filterAction = WorkerInPath::Veto;
      } else if (name[0] == '-' or name[0] == '+') {
        filterAction = WorkerInPath::Ignore;
      }
      if (name[0] == '|' or name[0] == '+') {
        //cms.wait was specified so do not run concurrently
        doNotRunConcurrently = true;
      }

      std::string moduleLabel = name;
      if (filterAction != WorkerInPath::Normal or name[0] == '|') {
        moduleLabel.erase(0, 1);
      }

      Worker* worker = getWorker(moduleLabel, proc_pset, workerManager_, preg, prealloc, processConfiguration);
      if (worker == nullptr) {
        std::string pathType("endpath");
        if (!search_all(endPathNames, pathName)) {
          pathType = std::string("path");
        }
        throw Exception(errors::Configuration)
            << "The unknown module label \"" << moduleLabel << "\" appears in " << pathType << " \"" << pathName
            << "\"\n please check spelling or remove that label from the path.";
      }

      if (ignoreFilters && filterAction != WorkerInPath::Ignore && worker->moduleType() == Worker::kFilter) {
        // We have a filter on an end path, and the filter is not explicitly ignored.
        // See if the filter is allowed.
        std::vector<std::string> allowed_filters = proc_pset.getUntrackedParameter<vstring>("@filters_on_endpaths");
        if (!search_all(allowed_filters, worker->description()->moduleName())) {
          // Filter is not allowed. Ignore the result, and issue a warning.
          filterAction = WorkerInPath::Ignore;
          LogWarning("FilterOnEndPath") << "The EDFilter '" << worker->description()->moduleName()
                                        << "' with module label '" << moduleLabel << "' appears on EndPath '"
                                        << pathName << "'.\n"
                                        << "The return value of the filter will be ignored.\n"
                                        << "To suppress this warning, either remove the filter from the endpath,\n"
                                        << "or explicitly ignore it in the configuration by using cms.ignore().\n";
        }
      }
      bool runConcurrently = not doNotRunConcurrently;
      if (runConcurrently && worker->moduleType() == Worker::kFilter and filterAction != WorkerInPath::Ignore) {
        runConcurrently = false;
      }

      auto condModules = tryToPlaceConditionalModules(worker,
                                                      conditionalmods,
                                                      conditionalModsBranches,
                                                      conditionalTaskHelper.aliasMap(),
                                                      proc_pset,
                                                      preg,
                                                      prealloc,
                                                      processConfiguration);
      for (auto condMod : condModules) {
        tmpworkers.emplace_back(
            condMod, WorkerInPath::Ignore, conditionalModOrder[condMod->description()->moduleLabel()], true);
      }

      tmpworkers.emplace_back(worker, filterAction, placeInPath, runConcurrently);
      ++placeInPath;
    }

    out.swap(tmpworkers);
  }

  void StreamSchedule::fillTrigPath(ParameterSet& proc_pset,
                                    ProductRegistry& preg,
                                    PreallocationConfiguration const* prealloc,
                                    std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                    int bitpos,
                                    std::string const& name,
                                    TrigResPtr trptr,
                                    std::vector<std::string> const& endPathNames,
                                    ConditionalTaskHelper const& conditionalTaskHelper) {
    PathWorkers tmpworkers;
    fillWorkers(
        proc_pset, preg, prealloc, processConfiguration, name, false, tmpworkers, endPathNames, conditionalTaskHelper);

    // an empty path will cause an extra bit that is not used
    if (!tmpworkers.empty()) {
      trig_paths_.emplace_back(bitpos,
                               name,
                               tmpworkers,
                               trptr,
                               actionTable(),
                               actReg_,
                               &streamContext_,
                               &skippingEvent_,
                               PathContext::PathType::kPath);
    } else {
      empty_trig_paths_.push_back(bitpos);
    }
    for (WorkerInPath const& workerInPath : tmpworkers) {
      addToAllWorkers(workerInPath.getWorker());
    }
  }

  void StreamSchedule::fillEndPath(ParameterSet& proc_pset,
                                   ProductRegistry& preg,
                                   PreallocationConfiguration const* prealloc,
                                   std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                   int bitpos,
                                   std::string const& name,
                                   std::vector<std::string> const& endPathNames,
                                   ConditionalTaskHelper const& conditionalTaskHelper) {
    PathWorkers tmpworkers;
    fillWorkers(
        proc_pset, preg, prealloc, processConfiguration, name, true, tmpworkers, endPathNames, conditionalTaskHelper);

    if (!tmpworkers.empty()) {
      //EndPaths are not supposed to stop if SkipEvent type exception happens
      end_paths_.emplace_back(bitpos,
                              name,
                              tmpworkers,
                              TrigResPtr(),
                              actionTable(),
                              actReg_,
                              &streamContext_,
                              nullptr,
                              PathContext::PathType::kEndPath);
    } else {
      empty_end_paths_.push_back(bitpos);
    }
    for (WorkerInPath const& workerInPath : tmpworkers) {
      addToAllWorkers(workerInPath.getWorker());
    }
  }

  void StreamSchedule::beginStream() { workerManager_.beginStream(streamID_, streamContext_); }

  void StreamSchedule::endStream() { workerManager_.endStream(streamID_, streamContext_); }

  void StreamSchedule::replaceModule(maker::ModuleHolder* iMod, std::string const& iLabel) {
    Worker* found = nullptr;
    for (auto const& worker : allWorkers()) {
      if (worker->description()->moduleLabel() == iLabel) {
        found = worker;
        break;
      }
    }
    if (nullptr == found) {
      return;
    }

    iMod->replaceModuleFor(found);
    found->beginStream(streamID_, streamContext_);
  }

  void StreamSchedule::deleteModule(std::string const& iLabel) { workerManager_.deleteModuleIfExists(iLabel); }

  std::vector<ModuleDescription const*> StreamSchedule::getAllModuleDescriptions() const {
    std::vector<ModuleDescription const*> result;
    result.reserve(allWorkers().size());

    for (auto const& worker : allWorkers()) {
      ModuleDescription const* p = worker->description();
      result.push_back(p);
    }
    return result;
  }

  void StreamSchedule::processOneEventAsync(
      WaitingTaskHolder iTask,
      EventTransitionInfo& info,
      ServiceToken const& serviceToken,
      std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters) {
    EventPrincipal& ep = info.principal();

    // Caught exception is propagated via WaitingTaskHolder
    CMS_SA_ALLOW try {
      this->resetAll();

      using Traits = OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>;

      Traits::setStreamContext(streamContext_, ep);
      //a service may want to communicate with another service
      ServiceRegistry::Operate guard(serviceToken);
      Traits::preScheduleSignal(actReg_.get(), &streamContext_);

      // Data dependencies need to be set up before marking empty
      // (End)Paths complete in case something consumes the status of
      // the empty (EndPath)
      workerManager_.setupResolvers(ep);
      workerManager_.setupOnDemandSystem(info);

      HLTPathStatus hltPathStatus(hlt::Pass, 0);
      for (int empty_trig_path : empty_trig_paths_) {
        results_->at(empty_trig_path) = hltPathStatus;
        pathStatusInserters[empty_trig_path]->setPathStatus(streamID_, hltPathStatus);
        std::exception_ptr except = pathStatusInserterWorkers_[empty_trig_path]
                                        ->runModuleDirectly<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(
                                            info, streamID_, ParentContext(&streamContext_), &streamContext_);
        if (except) {
          iTask.doneWaiting(except);
          return;
        }
      }
      for (int empty_end_path : empty_end_paths_) {
        std::exception_ptr except = endPathStatusInserterWorkers_[empty_end_path]
                                        ->runModuleDirectly<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(
                                            info, streamID_, ParentContext(&streamContext_), &streamContext_);
        if (except) {
          iTask.doneWaiting(except);
          return;
        }
      }

      ++total_events_;

      //use to give priorities on an error to ones from Paths
      auto pathErrorHolder = std::make_unique<std::atomic<std::exception_ptr*>>(nullptr);
      auto pathErrorPtr = pathErrorHolder.get();
      ServiceWeakToken weakToken = serviceToken;
      auto allPathsDone = make_waiting_task(
          [iTask, this, weakToken, pathError = std::move(pathErrorHolder)](std::exception_ptr const* iPtr) mutable {
            ServiceRegistry::Operate operate(weakToken.lock());

            std::exception_ptr ptr;
            if (pathError->load()) {
              ptr = *pathError->load();
              delete pathError->load();
            }
            if ((not ptr) and iPtr) {
              ptr = *iPtr;
            }
            iTask.doneWaiting(finishProcessOneEvent(ptr));
          });
      //The holder guarantees that if the paths finish before the loop ends
      // that we do not start too soon. It also guarantees that the task will
      // run under that condition.
      WaitingTaskHolder allPathsHolder(*iTask.group(), allPathsDone);

      auto pathsDone = make_waiting_task([allPathsHolder, pathErrorPtr, transitionInfo = info, this, weakToken](
                                             std::exception_ptr const* iPtr) mutable {
        ServiceRegistry::Operate operate(weakToken.lock());

        if (iPtr) {
          //this is used to prioritize this error over one
          // that happens in EndPath or Accumulate
          pathErrorPtr->store(new std::exception_ptr(*iPtr));
        }
        finishedPaths(*pathErrorPtr, std::move(allPathsHolder), transitionInfo);
      });

      //The holder guarantees that if the paths finish before the loop ends
      // that we do not start too soon. It also guarantees that the task will
      // run under that condition.
      WaitingTaskHolder taskHolder(*iTask.group(), pathsDone);

      //start end paths first so on single threaded the paths will run first
      WaitingTaskHolder hAllPathsDone(*iTask.group(), allPathsDone);
      for (auto it = end_paths_.rbegin(), itEnd = end_paths_.rend(); it != itEnd; ++it) {
        it->processOneOccurrenceAsync(hAllPathsDone, info, serviceToken, streamID_, &streamContext_);
      }

      for (auto it = trig_paths_.rbegin(), itEnd = trig_paths_.rend(); it != itEnd; ++it) {
        it->processOneOccurrenceAsync(taskHolder, info, serviceToken, streamID_, &streamContext_);
      }

      ParentContext parentContext(&streamContext_);
      workerManager_.processAccumulatorsAsync<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(
          hAllPathsDone, info, serviceToken, streamID_, parentContext, &streamContext_);
    } catch (...) {
      iTask.doneWaiting(std::current_exception());
    }
  }

  void StreamSchedule::finishedPaths(std::atomic<std::exception_ptr*>& iExcept,
                                     WaitingTaskHolder iWait,
                                     EventTransitionInfo& info) {
    if (iExcept) {
      // Caught exception is propagated via WaitingTaskHolder
      CMS_SA_ALLOW try { std::rethrow_exception(*(iExcept.load())); } catch (cms::Exception& e) {
        exception_actions::ActionCodes action = actionTable().find(e.category());
        assert(action != exception_actions::IgnoreCompletely);
        assert(action != exception_actions::FailPath);
        if (action == exception_actions::SkipEvent) {
          edm::printCmsExceptionWarning("SkipEvent", e);
          *(iExcept.load()) = std::exception_ptr();
        } else {
          *(iExcept.load()) = std::current_exception();
        }
      } catch (...) {
        *(iExcept.load()) = std::current_exception();
      }
    }

    if ((not iExcept) and results_->accept()) {
      ++total_passed_;
    }

    if (nullptr != results_inserter_.get()) {
      // Caught exception is propagated to the caller
      CMS_SA_ALLOW try {
        //Even if there was an exception, we need to allow results inserter
        // to run since some module may be waiting on its results.
        ParentContext parentContext(&streamContext_);
        using Traits = OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>;

        auto expt = results_inserter_->runModuleDirectly<Traits>(info, streamID_, parentContext, &streamContext_);
        if (expt) {
          std::rethrow_exception(expt);
        }
      } catch (cms::Exception& ex) {
        if (not iExcept) {
          if (ex.context().empty()) {
            std::ostringstream ost;
            ost << "Processing Event " << info.principal().id();
            ex.addContext(ost.str());
          }
          iExcept.store(new std::exception_ptr(std::current_exception()));
        }
      } catch (...) {
        if (not iExcept) {
          iExcept.store(new std::exception_ptr(std::current_exception()));
        }
      }
    }
    std::exception_ptr ptr;
    if (iExcept) {
      ptr = *iExcept.load();
    }
    iWait.doneWaiting(ptr);
  }

  std::exception_ptr StreamSchedule::finishProcessOneEvent(std::exception_ptr iExcept) {
    using Traits = OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>;

    if (iExcept) {
      //add context information to the exception and print message
      try {
        convertException::wrap([&]() { std::rethrow_exception(iExcept); });
      } catch (cms::Exception& ex) {
        bool const cleaningUpAfterException = false;
        if (ex.context().empty()) {
          addContextAndPrintException("Calling function StreamSchedule::processOneEvent", ex, cleaningUpAfterException);
        } else {
          addContextAndPrintException("", ex, cleaningUpAfterException);
        }
        iExcept = std::current_exception();
      }

      actReg_->preStreamEarlyTerminationSignal_(streamContext_, TerminationOrigin::ExceptionFromThisContext);
    }
    // Caught exception is propagated to the caller
    CMS_SA_ALLOW try { Traits::postScheduleSignal(actReg_.get(), &streamContext_); } catch (...) {
      if (not iExcept) {
        iExcept = std::current_exception();
      }
    }
    if (not iExcept) {
      resetEarlyDelete();
    }

    return iExcept;
  }

  void StreamSchedule::availablePaths(std::vector<std::string>& oLabelsToFill) const {
    oLabelsToFill.reserve(trig_paths_.size());
    std::transform(trig_paths_.begin(),
                   trig_paths_.end(),
                   std::back_inserter(oLabelsToFill),
                   std::bind(&Path::name, std::placeholders::_1));
  }

  void StreamSchedule::modulesInPath(std::string const& iPathLabel, std::vector<std::string>& oLabelsToFill) const {
    TrigPaths::const_iterator itFound = std::find_if(
        trig_paths_.begin(),
        trig_paths_.end(),
        std::bind(std::equal_to<std::string>(), iPathLabel, std::bind(&Path::name, std::placeholders::_1)));
    if (itFound != trig_paths_.end()) {
      oLabelsToFill.reserve(itFound->size());
      for (size_t i = 0; i < itFound->size(); ++i) {
        oLabelsToFill.push_back(itFound->getWorker(i)->description()->moduleLabel());
      }
    }
  }

  void StreamSchedule::moduleDescriptionsInPath(std::string const& iPathLabel,
                                                std::vector<ModuleDescription const*>& descriptions,
                                                unsigned int hint) const {
    descriptions.clear();
    bool found = false;
    TrigPaths::const_iterator itFound;

    if (hint < trig_paths_.size()) {
      itFound = trig_paths_.begin() + hint;
      if (itFound->name() == iPathLabel)
        found = true;
    }
    if (!found) {
      // if the hint did not work, do it the slow way
      itFound = std::find_if(
          trig_paths_.begin(),
          trig_paths_.end(),
          std::bind(std::equal_to<std::string>(), iPathLabel, std::bind(&Path::name, std::placeholders::_1)));
      if (itFound != trig_paths_.end())
        found = true;
    }
    if (found) {
      descriptions.reserve(itFound->size());
      for (size_t i = 0; i < itFound->size(); ++i) {
        descriptions.push_back(itFound->getWorker(i)->description());
      }
    }
  }

  void StreamSchedule::moduleDescriptionsInEndPath(std::string const& iEndPathLabel,
                                                   std::vector<ModuleDescription const*>& descriptions,
                                                   unsigned int hint) const {
    descriptions.clear();
    bool found = false;
    TrigPaths::const_iterator itFound;

    if (hint < end_paths_.size()) {
      itFound = end_paths_.begin() + hint;
      if (itFound->name() == iEndPathLabel)
        found = true;
    }
    if (!found) {
      // if the hint did not work, do it the slow way
      itFound = std::find_if(
          end_paths_.begin(),
          end_paths_.end(),
          std::bind(std::equal_to<std::string>(), iEndPathLabel, std::bind(&Path::name, std::placeholders::_1)));
      if (itFound != end_paths_.end())
        found = true;
    }
    if (found) {
      descriptions.reserve(itFound->size());
      for (size_t i = 0; i < itFound->size(); ++i) {
        descriptions.push_back(itFound->getWorker(i)->description());
      }
    }
  }

  static void fillModuleInPathSummary(Path const& path, size_t which, ModuleInPathSummary& sum) {
    sum.timesVisited += path.timesVisited(which);
    sum.timesPassed += path.timesPassed(which);
    sum.timesFailed += path.timesFailed(which);
    sum.timesExcept += path.timesExcept(which);
    sum.moduleLabel = path.getWorker(which)->description()->moduleLabel();
    sum.bitPosition = path.bitPosition(which);
  }

  static void fillPathSummary(Path const& path, PathSummary& sum) {
    sum.name = path.name();
    sum.bitPosition = path.bitPosition();
    sum.timesRun += path.timesRun();
    sum.timesPassed += path.timesPassed();
    sum.timesFailed += path.timesFailed();
    sum.timesExcept += path.timesExcept();

    Path::size_type sz = path.size();
    if (sum.moduleInPathSummaries.empty()) {
      std::vector<ModuleInPathSummary> temp(sz);
      for (size_t i = 0; i != sz; ++i) {
        fillModuleInPathSummary(path, i, temp[i]);
      }
      sum.moduleInPathSummaries.swap(temp);
    } else {
      assert(sz == sum.moduleInPathSummaries.size());
      for (size_t i = 0; i != sz; ++i) {
        fillModuleInPathSummary(path, i, sum.moduleInPathSummaries[i]);
      }
    }
  }

  static void fillWorkerSummaryAux(Worker const& w, WorkerSummary& sum) {
    sum.timesVisited += w.timesVisited();
    sum.timesRun += w.timesRun();
    sum.timesPassed += w.timesPassed();
    sum.timesFailed += w.timesFailed();
    sum.timesExcept += w.timesExcept();
    sum.moduleLabel = w.description()->moduleLabel();
  }

  static void fillWorkerSummary(Worker const* pw, WorkerSummary& sum) { fillWorkerSummaryAux(*pw, sum); }

  void StreamSchedule::getTriggerReport(TriggerReport& rep) const {
    rep.eventSummary.totalEvents += totalEvents();
    rep.eventSummary.totalEventsPassed += totalEventsPassed();
    rep.eventSummary.totalEventsFailed += totalEventsFailed();

    fill_summary(trig_paths_, rep.trigPathSummaries, &fillPathSummary);
    fill_summary(end_paths_, rep.endPathSummaries, &fillPathSummary);
    fill_summary(allWorkers(), rep.workerSummaries, &fillWorkerSummary);
  }

  void StreamSchedule::clearCounters() {
    using std::placeholders::_1;
    total_events_ = total_passed_ = 0;
    for_all(trig_paths_, std::bind(&Path::clearCounters, _1));
    for_all(end_paths_, std::bind(&Path::clearCounters, _1));
    for_all(allWorkers(), std::bind(&Worker::clearCounters, _1));
  }

  void StreamSchedule::resetAll() {
    skippingEvent_ = false;
    results_->reset();
  }

  void StreamSchedule::addToAllWorkers(Worker* w) { workerManager_.addToAllWorkers(w); }

  void StreamSchedule::resetEarlyDelete() {
    //must be sure we have cleared the count first
    for (auto& count : earlyDeleteBranchToCount_) {
      count.count = 0;
    }
    //now reset based on how many helpers use that branch
    for (auto& index : earlyDeleteHelperToBranchIndicies_) {
      ++(earlyDeleteBranchToCount_[index].count);
    }
    for (auto& helper : earlyDeleteHelpers_) {
      helper.reset();
    }
  }

  void StreamSchedule::makePathStatusInserters(
      std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters,
      std::vector<edm::propagate_const<std::shared_ptr<EndPathStatusInserter>>>& endPathStatusInserters,
      ExceptionToActionTable const& actions) {
    int bitpos = 0;
    unsigned int indexEmpty = 0;
    unsigned int indexOfPath = 0;
    for (auto& pathStatusInserter : pathStatusInserters) {
      std::shared_ptr<PathStatusInserter> inserterPtr = get_underlying(pathStatusInserter);
      WorkerPtr workerPtr(
          new edm::WorkerT<PathStatusInserter::ModuleType>(inserterPtr, inserterPtr->moduleDescription(), &actions));
      pathStatusInserterWorkers_.emplace_back(workerPtr);
      workerPtr->setActivityRegistry(actReg_);
      addToAllWorkers(workerPtr.get());

      // A little complexity here because a C++ Path object is not
      // instantiated and put into end_paths if there are no modules
      // on the configured path.
      if (indexEmpty < empty_trig_paths_.size() && bitpos == empty_trig_paths_.at(indexEmpty)) {
        ++indexEmpty;
      } else {
        trig_paths_.at(indexOfPath).setPathStatusInserter(inserterPtr.get(), workerPtr.get());
        ++indexOfPath;
      }
      ++bitpos;
    }

    bitpos = 0;
    indexEmpty = 0;
    indexOfPath = 0;
    for (auto& endPathStatusInserter : endPathStatusInserters) {
      std::shared_ptr<EndPathStatusInserter> inserterPtr = get_underlying(endPathStatusInserter);
      WorkerPtr workerPtr(
          new edm::WorkerT<EndPathStatusInserter::ModuleType>(inserterPtr, inserterPtr->moduleDescription(), &actions));
      endPathStatusInserterWorkers_.emplace_back(workerPtr);
      workerPtr->setActivityRegistry(actReg_);
      addToAllWorkers(workerPtr.get());

      // A little complexity here because a C++ Path object is not
      // instantiated and put into end_paths if there are no modules
      // on the configured path.
      if (indexEmpty < empty_end_paths_.size() && bitpos == empty_end_paths_.at(indexEmpty)) {
        ++indexEmpty;
      } else {
        end_paths_.at(indexOfPath).setPathStatusInserter(nullptr, workerPtr.get());
        ++indexOfPath;
      }
      ++bitpos;
    }
  }
}  // namespace edm
