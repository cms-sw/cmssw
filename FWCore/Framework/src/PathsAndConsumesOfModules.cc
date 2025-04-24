#include "FWCore/Framework/interface/PathsAndConsumesOfModules.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/ESModuleProducesInfo.h"
#include "FWCore/Framework/interface/ESModuleConsumesMinimalInfo.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/ServiceRegistry/interface/ESModuleConsumesInfo.h"
#include "FWCore/ServiceRegistry/interface/ModuleConsumesESInfo.h"
#include "FWCore/ServiceRegistry/interface/ModuleConsumesInfo.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include <algorithm>
#include <limits>
#include <unordered_set>
#include <utility>
#include <set>

#include <iostream>  // for debugging

namespace edm {

  namespace {
    void insertFoundModuleLabel(edm::KindOfType consumedTypeKind,
                                edm::TypeID consumedType,
                                const char* consumedModuleLabel,
                                const char* consumedProductInstance,
                                std::vector<ModuleDescription const*>& modules,
                                std::set<std::string>& alreadyFound,
                                std::map<std::string, ModuleDescription const*> const& labelsToDesc,
                                ProductRegistry const& preg) {
      // Convert from label string to module description, eliminate duplicates,
      // then insert into the vector of modules
      if (auto it = labelsToDesc.find(consumedModuleLabel); it != labelsToDesc.end()) {
        if (alreadyFound.insert(consumedModuleLabel).second) {
          modules.push_back(it->second);
        }
        return;
      }
      // Deal with EDAlias's by converting to the original module label first
      if (auto aliasToModuleLabels =
              preg.aliasToModules(consumedTypeKind, consumedType, consumedModuleLabel, consumedProductInstance);
          not aliasToModuleLabels.empty()) {
        bool foundInLabelsToDesc = false;
        for (auto const& label : aliasToModuleLabels) {
          if (auto it = labelsToDesc.find(label); it != labelsToDesc.end()) {
            if (alreadyFound.insert(label).second) {
              modules.push_back(it->second);
            }
            foundInLabelsToDesc = true;
          } else {
            if (label == "source") {
              foundInLabelsToDesc = true;
            }
          }
        }
        if (foundInLabelsToDesc) {
          return;
        }
      }
      // Ignore the source products, we are only interested in module products.
      // As far as I know, it should never be anything else so throw if something
      // unknown gets passed in.
      if (std::string_view(consumedModuleLabel) != "source") {
        throw cms::Exception("EDConsumerBase", "insertFoundModuleLabel")
            << "Couldn't find ModuleDescription for the consumed product type: '" << consumedType.className()
            << "' module label: '" << consumedModuleLabel << "' product instance name: '" << consumedProductInstance
            << "'";
      }
    }

    void modulesWhoseProductsAreConsumed(Worker* iWorker,
                                         std::array<std::vector<ModuleDescription const*>*, NumBranchTypes>& modulesAll,
                                         ProductRegistry const& preg,
                                         std::map<std::string, ModuleDescription const*> const& labelsToDesc,
                                         std::string const& processName) {
      std::set<std::string> alreadyFound;

      for (ModuleConsumesInfo const& consumesInfo : iWorker->moduleConsumesInfos()) {
        ProductResolverIndexHelper const& helper = *preg.productLookup(consumesInfo.branchType());
        std::vector<ModuleDescription const*>& modules = *modulesAll[consumesInfo.branchType()];

        auto consumedModuleLabel = consumesInfo.label();
        auto consumedProductInstance = consumesInfo.instance();
        auto consumedProcessName = consumesInfo.process();
        auto kind = consumesInfo.kindOfType();
        auto const& typeID = consumesInfo.type();

        if (not consumesInfo.skipCurrentProcess()) {
          assert(*consumedModuleLabel.data() !=
                 '\0');  // consumesMany used to create empty labels before we removed consumesMany
          if (*consumedProcessName.data() != '\0') {  // process name is specified in consumes call
            if (helper.index(kind,
                             typeID,
                             consumedModuleLabel.data(),
                             consumedProductInstance.data(),
                             consumedProcessName.data()) != ProductResolverIndexInvalid) {
              if (processName == consumedProcessName) {
                insertFoundModuleLabel(kind,
                                       typeID,
                                       consumedModuleLabel.data(),
                                       consumedProductInstance.data(),
                                       modules,
                                       alreadyFound,
                                       labelsToDesc,
                                       preg);
              }
            }
          } else {  // process name was empty
            auto matches =
                helper.relatedIndexes(kind, typeID, consumedModuleLabel.data(), consumedProductInstance.data());
            for (unsigned int j = 0; j < matches.numberOfMatches(); ++j) {
              if (processName == matches.processName(j)) {
                insertFoundModuleLabel(kind,
                                       typeID,
                                       consumedModuleLabel.data(),
                                       consumedProductInstance.data(),
                                       modules,
                                       alreadyFound,
                                       labelsToDesc,
                                       preg);
              }
            }
          }
        }
      };
    }
    void fillModuleAndConsumesInfo(Schedule::AllWorkers const& allWorkers,
                                   std::vector<ModuleDescription const*>& allModuleDescriptions,
                                   std::vector<std::pair<unsigned int, unsigned int>>& moduleIDToIndex,
                                   std::array<std::vector<std::vector<ModuleDescription const*>>, NumBranchTypes>&
                                       modulesWhoseProductsAreConsumedBy,
                                   ProductRegistry const& preg) {
      allModuleDescriptions.clear();
      moduleIDToIndex.clear();
      for (auto iBranchType = 0U; iBranchType < NumBranchTypes; ++iBranchType) {
        modulesWhoseProductsAreConsumedBy[iBranchType].clear();
      }

      allModuleDescriptions.reserve(allWorkers.size());
      moduleIDToIndex.reserve(allWorkers.size());
      for (auto iBranchType = 0U; iBranchType < NumBranchTypes; ++iBranchType) {
        modulesWhoseProductsAreConsumedBy[iBranchType].resize(allWorkers.size());
      }

      std::map<std::string, ModuleDescription const*> labelToDesc;
      unsigned int i = 0;
      for (auto const& worker : allWorkers) {
        ModuleDescription const* p = worker->description();
        allModuleDescriptions.push_back(p);
        moduleIDToIndex.push_back(std::pair<unsigned int, unsigned int>(p->id(), i));
        labelToDesc[p->moduleLabel()] = p;
        ++i;
      }
      sort_all(moduleIDToIndex);

      i = 0;
      for (auto const& worker : allWorkers) {
        std::array<std::vector<ModuleDescription const*>*, NumBranchTypes> modules;
        for (auto iBranchType = 0U; iBranchType < NumBranchTypes; ++iBranchType) {
          modules[iBranchType] = &modulesWhoseProductsAreConsumedBy[iBranchType].at(i);
        }
        try {
          modulesWhoseProductsAreConsumed(worker, modules, preg, labelToDesc, worker->description()->processName());
        } catch (cms::Exception& ex) {
          ex.addContext("Calling Worker::modulesWhoseProductsAreConsumed() for module " +
                        worker->description()->moduleLabel());
          throw;
        }
        ++i;
      }
    }
  }  // namespace

  void PathsAndConsumesOfModules::initialize(Schedule const* schedule, std::shared_ptr<ProductRegistry const> preg) {
    schedule_ = schedule;
    preg_ = preg;

    paths_.clear();
    schedule->triggerPaths(paths_);

    endPaths_.clear();
    schedule->endPaths(endPaths_);

    modulesOnPaths_.resize(paths_.size());
    unsigned int i = 0;
    unsigned int hint = 0;
    for (auto const& path : paths_) {
      schedule->moduleDescriptionsInPath(path, modulesOnPaths_.at(i), hint);
      if (!modulesOnPaths_.at(i).empty())
        ++hint;
      ++i;
    }

    modulesOnEndPaths_.resize(endPaths_.size());
    i = 0;
    hint = 0;
    for (auto const& endpath : endPaths_) {
      schedule->moduleDescriptionsInEndPath(endpath, modulesOnEndPaths_.at(i), hint);
      if (!modulesOnEndPaths_.at(i).empty())
        ++hint;
      ++i;
    }

    fillModuleAndConsumesInfo(
        schedule_->allWorkers(), allModuleDescriptions_, moduleIDToIndex_, modulesWhoseProductsAreConsumedBy_, *preg);
  }

  using ProducedByESModule = PathsAndConsumesOfModules::ProducedByESModule;
  namespace {
    void esModulesWhoseProductsAreConsumed(
        Worker* worker,
        std::array<std::vector<eventsetup::ComponentDescription const*>*, kNumberOfEventSetupTransitions>& esModules,
        ProducedByESModule const& producedByESModule) {
      std::array<std::set<std::string>, kNumberOfEventSetupTransitions> alreadyFound;

      for (auto const& info : worker->moduleConsumesMinimalESInfos()) {
        auto const& recordInfo = producedByESModule.find(info.record_);
        if (recordInfo != producedByESModule.end()) {
          auto itFound = recordInfo->second.find(info.dataKey_);
          if (itFound != recordInfo->second.end()) {
            auto const& componentDescription = itFound->second.componentDescription_;
            if (componentDescription) {
              std::string const& moduleLabel =
                  componentDescription->label_.empty() ? componentDescription->type_ : componentDescription->label_;
              //check for matching labels if required
              if (info.componentLabel_.empty() || info.componentLabel_ == moduleLabel) {
                auto transitionIndex = static_cast<unsigned int>(info.transition_);
                if (alreadyFound[transitionIndex].insert(moduleLabel).second) {
                  esModules[transitionIndex]->push_back(componentDescription);
                }
              }
            }
          }
        }
      }
    }

    std::array<std::vector<std::vector<eventsetup::ComponentDescription const*>>, kNumberOfEventSetupTransitions>
    esModulesWhoseProductsAreConsumedByCreate(Schedule::AllWorkers const& allWorkers,
                                              ProducedByESModule const& producedByESModule) {
      std::array<std::vector<std::vector<eventsetup::ComponentDescription const*>>, kNumberOfEventSetupTransitions>
          esModulesWhoseProductsAreConsumedBy;

      for (auto& item : esModulesWhoseProductsAreConsumedBy) {
        item.resize(allWorkers.size());
      }

      for (unsigned int i = 0; auto const& worker : allWorkers) {
        std::array<std::vector<eventsetup::ComponentDescription const*>*, kNumberOfEventSetupTransitions> esModules;
        for (auto transition = 0U; transition < kNumberOfEventSetupTransitions; ++transition) {
          esModules[transition] = &esModulesWhoseProductsAreConsumedBy[transition].at(i);
        }
        try {
          esModulesWhoseProductsAreConsumed(worker, esModules, producedByESModule);
        } catch (cms::Exception& ex) {
          ex.addContext("Calling Worker::esModulesWhoseProductsAreConsumed() for module " +
                        worker->description()->moduleLabel());
          throw;
        }
        ++i;
      }
      return esModulesWhoseProductsAreConsumedBy;
    }

    ProducedByESModule fillProducedByESModule(eventsetup::EventSetupProvider const& esProvider) {
      ProducedByESModule producedByESModule;

      std::set<eventsetup::EventSetupRecordKey> keys;
      esProvider.fillKeys(keys);

      for (auto const& recordKey : keys) {
        auto const* providers = esProvider.tryToGetRecordProvider(recordKey);
        if (providers) {
          auto const& datakeys = providers->registeredDataKeys();
          auto const& componentsForDataKeys = providers->componentsForRegisteredDataKeys();
          auto const& produceMethodIDs = providers->produceMethodIDsForRegisteredDataKeys();
          assert(datakeys.size() == componentsForDataKeys.size());
          assert(datakeys.size() == produceMethodIDs.size());
          for (unsigned int i = 0; i < datakeys.size(); ++i) {
            auto const& dataKey = datakeys[i];
            auto const* componentDescription = componentsForDataKeys[i];
            auto produceMethodID = produceMethodIDs[i];
            producedByESModule[recordKey][dataKey] = {componentDescription, produceMethodID};
          }
        }
      }
      return producedByESModule;
    }

    std::vector<std::vector<eventsetup::ComponentDescription const*>> esModulesWhoseProductsAreConsumedByESModuleCreate(
        std::vector<const eventsetup::ESProductResolverProvider*> const& allESProductResolverProviders,
        ProducedByESModule const& producedByESModule) {
      std::vector<std::vector<eventsetup::ComponentDescription const*>> retValue;

      retValue.resize(allESProductResolverProviders.size());
      auto it = retValue.begin();
      for (auto& provider : allESProductResolverProviders) {
        ESProducer const* esProducer = dynamic_cast<ESProducer const*>(provider);
        if (esProducer) {
          std::set<unsigned int> alreadyFound;
          auto const& consumesInfo = esProducer->esModuleConsumesMinimalInfos();
          for (auto const& info : consumesInfo) {
            auto const& recordKey = info.recordForDataKey_;
            auto const& dataKey = info.dataKey_;
            auto itFound = producedByESModule.find(recordKey);
            if (itFound != producedByESModule.end()) {
              if (dataKey.name() == "@mayConsume") {
                // This is a "may consume" case, we need to find all components that may produce this dataKey
                for (auto const& [dataKey, produceInfo] : itFound->second) {
                  auto componentDescription = produceInfo.componentDescription_;
                  if (dataKey.type() == info.dataKey_.type()) {
                    if (componentDescription and alreadyFound.find(componentDescription->id_) == alreadyFound.end()) {
                      alreadyFound.insert(componentDescription->id_);
                      it->push_back(componentDescription);
                    }
                  }
                }
              } else {
                // This is a normal case, we need to find the specific component that produces this dataKey
                auto itDataKey = itFound->second.find(dataKey);
                if (itDataKey != itFound->second.end()) {
                  eventsetup::ComponentDescription const* componentDescription =
                      itDataKey->second.componentDescription_;
                  if (componentDescription and alreadyFound.find(componentDescription->id_) == alreadyFound.end()) {
                    //an empty label matches any label, else we need an exact match
                    if (info.componentLabel_.empty() || info.componentLabel_ == componentDescription->label_) {
                      alreadyFound.insert(componentDescription->id_);
                      it->push_back(componentDescription);
                    }
                  }
                }
              }
            }
          }
        }
        ++it;
      }
      return retValue;
    }

  }  // namespace

  void PathsAndConsumesOfModules::initializeForEventSetup(eventsetup::EventSetupProvider const& eventSetupProvider) {
    eventSetupProvider.fillAllESProductResolverProviders(allESProductResolverProviders_);

    producedByESModule_ = fillProducedByESModule(eventSetupProvider);

    esModulesWhoseProductsAreConsumedBy_ =
        esModulesWhoseProductsAreConsumedByCreate(schedule_->allWorkers(), producedByESModule_);

    for (unsigned int i = 0; i < allESProductResolverProviders_.size(); ++i) {
      eventsetup::ComponentDescription const& componentDescription = allESProductResolverProviders_[i]->description();
      esModuleIDToIndex_.emplace_back(componentDescription.id_, i);
      allComponentDescriptions_.push_back(&componentDescription);
    }
    sort_all(esModuleIDToIndex_);

    esModulesWhoseProductsAreConsumedByESModule_ =
        esModulesWhoseProductsAreConsumedByESModuleCreate(allESProductResolverProviders_, producedByESModule_);
    eventSetupInfoInitialized_ = true;
  }

  void PathsAndConsumesOfModules::checkEventSetupInitialization() const {
    // It is our intent to eventually migrate all Services using PathsAndConsumesOfModules
    // to use the LookupInitializationComplete signal and eliminate that argument from
    // the interface of the functions called for preBeginRun. Then everything related to
    // this function can be deleted.
    if (!eventSetupInfoInitialized_) {
      throw cms::Exception("LogicError")
          << "In PathsAndConsumesOfModules, a function used to access EventSetup information\n"
             "was called before the EventSetup information was initialized. The most likely\n"
             "fix for this is for the Service trying to access the information to use the\n"
             "LookupInitializationComplete signal instead of the PreBeginJob signal to get\n"
             "access to the PathsAndConsumesOfModules object. The EventSetup information is\n"
             "not initialized yet at preBeginJob.\n";
    }
  }

  void PathsAndConsumesOfModules::removeModules(std::vector<ModuleDescription const*> const& modules) {
    // First check that no modules on Paths are removed
    auto checkPath = [&modules](auto const& paths) {
      for (auto const& path : paths) {
        for (auto const& description : path) {
          if (std::find(modules.begin(), modules.end(), description) != modules.end()) {
            throw cms::Exception("Assert")
                << "PathsAndConsumesOfModules::removeModules() is trying to remove a module with label "
                << description->moduleLabel() << " id " << description->id() << " from a Path, this should not happen.";
          }
        }
      }
    };
    checkPath(modulesOnPaths_);
    checkPath(modulesOnEndPaths_);

    // Remove the modules and adjust the indices in idToIndex map
    for (auto iModule = 0U; iModule != allModuleDescriptions_.size(); ++iModule) {
      auto found = std::find(modules.begin(), modules.end(), allModuleDescriptions_[iModule]);
      if (found != modules.end()) {
        allModuleDescriptions_.erase(allModuleDescriptions_.begin() + iModule);
        for (auto iBranchType = 0U; iBranchType != NumBranchTypes; ++iBranchType) {
          modulesWhoseProductsAreConsumedBy_[iBranchType].erase(
              modulesWhoseProductsAreConsumedBy_[iBranchType].begin() + iModule);
        }
        for (auto& idToIndex : moduleIDToIndex_) {
          if (idToIndex.second >= iModule) {
            idToIndex.second--;
          }
        }
        --iModule;
      }
    }
  }

  std::vector<std::string> const& PathsAndConsumesOfModules::doPaths() const { return paths_; }
  std::vector<std::string> const& PathsAndConsumesOfModules::doEndPaths() const { return endPaths_; }

  std::vector<ModuleDescription const*> const& PathsAndConsumesOfModules::doAllModules() const {
    return allModuleDescriptions_;
  }

  ModuleDescription const* PathsAndConsumesOfModules::doModuleDescription(unsigned int moduleID) const {
    unsigned int dummy = 0;
    auto target = std::make_pair(moduleID, dummy);
    std::vector<std::pair<unsigned int, unsigned int>>::const_iterator iter =
        std::lower_bound(moduleIDToIndex_.begin(), moduleIDToIndex_.end(), target);
    if (iter == moduleIDToIndex_.end() || iter->first != moduleID) {
      throw Exception(errors::LogicError)
          << "PathsAndConsumesOfModules::moduleDescription: Unknown moduleID " << moduleID << "\n";
    }
    return allModuleDescriptions_.at(iter->second);
  }

  std::vector<ModuleDescription const*> const& PathsAndConsumesOfModules::doModulesOnPath(unsigned int pathIndex) const {
    return modulesOnPaths_.at(pathIndex);
  }

  std::vector<ModuleDescription const*> const& PathsAndConsumesOfModules::doModulesOnEndPath(
      unsigned int endPathIndex) const {
    return modulesOnEndPaths_.at(endPathIndex);
  }

  std::vector<ModuleDescription const*> const& PathsAndConsumesOfModules::doModulesWhoseProductsAreConsumedBy(
      unsigned int moduleID, BranchType branchType) const {
    return modulesWhoseProductsAreConsumedBy_[branchType].at(moduleIndex(moduleID));
  }

  std::vector<eventsetup::ComponentDescription const*> const&
  PathsAndConsumesOfModules::doESModulesWhoseProductsAreConsumedBy(unsigned int moduleID, Transition transition) const {
    checkEventSetupInitialization();
    return esModulesWhoseProductsAreConsumedBy_[static_cast<unsigned int>(transition)].at(moduleIndex(moduleID));
  }

  std::vector<ModuleConsumesInfo> PathsAndConsumesOfModules::doModuleConsumesInfos(unsigned int moduleID) const {
    Worker const* worker = schedule_->allWorkers().at(moduleIndex(moduleID));
    return worker->moduleConsumesInfos();
  }

  auto const& labelForComponentDescription(eventsetup::ComponentDescription const* description) {
    if (description->label_.empty()) {
      return description->type_;
    }
    return description->label_;
  }

  std::vector<ModuleConsumesESInfo> PathsAndConsumesOfModules::doModuleConsumesESInfos(unsigned int moduleID) const {
    checkEventSetupInitialization();
    Worker const* worker = schedule_->allWorkers().at(moduleIndex(moduleID));
    auto const& minConsumesESInfos = worker->moduleConsumesMinimalESInfos();
    std::vector<ModuleConsumesESInfo> result;
    result.reserve(minConsumesESInfos.size());
    for (auto const& minInfo : minConsumesESInfos) {
      ModuleConsumesESInfo info;
      info.eventSetupRecordType_ = minInfo.record_.name();
      info.productType_ = minInfo.dataKey_.type().name();
      //Moving this to a string_view is safe as the minInfo.dataKey_ does not own the memory
      info.productLabel_ = minInfo.dataKey_.name().value();
      info.requestedModuleLabel_ = minInfo.componentLabel_;
      info.transitionOfConsumer_ = minInfo.transition_;
      if (not info.requestedModuleLabel_.empty()) {
        auto itRec = producedByESModule_.find(minInfo.record_);
        if (itRec != producedByESModule_.end()) {
          auto itDataKeyInfo = itRec->second.find(minInfo.dataKey_);
          if (itDataKeyInfo != itRec->second.end()) {
            info.moduleLabelMismatch_ =
                labelForComponentDescription(itDataKeyInfo->second.componentDescription_) != info.requestedModuleLabel_;
          }
        }
      }

      // Initial values used in the case where there isn't an EventSetup
      // module to produce the requested data. Test whether moduleType
      // is empty to identify this case because it will be empty if and
      // only if this is true.
      info.moduleType_ = {};
      info.moduleLabel_ = {};
      info.produceMethodIDOfProducer_ = 0;
      info.isSource_ = false;
      info.isLooper_ = false;

      auto itRec = producedByESModule_.find(minInfo.record_);
      if (itRec != producedByESModule_.end()) {
        auto itDataKeyInfo = itRec->second.find(minInfo.dataKey_);
        if (itDataKeyInfo != itRec->second.end()) {
          auto produceMethodID = itDataKeyInfo->second.produceMethodID_;
          auto componentDescription = itDataKeyInfo->second.componentDescription_;
          if (componentDescription) {
            info.moduleType_ = componentDescription->type_;
            info.moduleLabel_ =
                componentDescription->label_.empty() ? componentDescription->type_ : componentDescription->label_;
            info.produceMethodIDOfProducer_ = produceMethodID;
            info.isSource_ = componentDescription->isSource_;
            info.isLooper_ = componentDescription->isLooper_;
          }
        }
      }
      result.emplace_back(info);
    };
    return result;
  }

  unsigned int PathsAndConsumesOfModules::doLargestModuleID() const {
    // moduleIDToIndex_ is sorted, so last element has the largest ID
    return moduleIDToIndex_.empty() ? 0 : moduleIDToIndex_.back().first;
  }

  std::vector<eventsetup::ComponentDescription const*> const& PathsAndConsumesOfModules::doAllESModules() const {
    checkEventSetupInitialization();
    return allComponentDescriptions_;
  }

  eventsetup::ComponentDescription const* PathsAndConsumesOfModules::doComponentDescription(
      unsigned int esModuleID) const {
    return allComponentDescriptions_.at(esModuleIndex(esModuleID));
  }

  std::vector<std::vector<eventsetup::ComponentDescription const*>> const&
  PathsAndConsumesOfModules::doESModulesWhoseProductsAreConsumedByESModule() const {
    checkEventSetupInitialization();
    return esModulesWhoseProductsAreConsumedByESModule_;
  }

  namespace {
    std::vector<std::vector<ESModuleConsumesInfo>> esModuleConsumesInfosCreate(
        ESProducer const& esProducer, ProducedByESModule const& producedByESModule) {
      auto const& consumesInfos = esProducer.esModuleConsumesMinimalInfos();
      std::vector<std::vector<ESModuleConsumesInfo>> result;
      // The outer vector has an entry per produce method ID
      unsigned int largestProduceMethodID = 0;
      for (auto const& produced : esProducer.producesInfo()) {
        if (produced.produceMethodID() > largestProduceMethodID) {
          largestProduceMethodID = produced.produceMethodID();
        }
      }
      result.resize(largestProduceMethodID + 1);
      if (consumesInfos.empty()) {
        return result;
      }
      result.resize(consumesInfos.back().produceMethodID_ + 1);

      for (auto const& esConsumesInfo : consumesInfos) {
        auto& resultForTransition = result[esConsumesInfo.produceMethodID_];

        ESModuleConsumesInfo info;
        info.produceMethodIDOfConsumer_ = esConsumesInfo.produceMethodID_;
        info.eventSetupRecordType_ = esConsumesInfo.recordForDataKey_.name();
        info.productType_ = esConsumesInfo.dataKey_.type().name();
        info.moduleType_ = {};
        info.moduleLabel_ = {};
        info.produceMethodIDOfProducer_ = 0;
        info.isSource_ = false;
        info.isLooper_ = false;
        info.moduleLabelMismatch_ = false;

        // If there is a chooser this is the special case of a "may consumes"
        if (esConsumesInfo.dataKey_.name() == "@mayConsume") {
          info.requestedModuleLabel_ = {};
          info.mayConsumes_ = true;
          info.mayConsumesFirstEntry_ = true;

          //look for matches
          auto itRec = producedByESModule.find(esConsumesInfo.recordForDataKey_);
          if (itRec == producedByESModule.end()) {
            // No producers for this record, so no products can be consumed
            info.productLabel_ = {};
            info.mayConsumesNoProducts_ = true;
            resultForTransition.push_back(info);
            continue;
          }
          // In the "may consumes" case, we iterate over all the possible data products
          // the EventSetup can produce with matching record type and product type.
          // With the current design of the mayConsumes feature, there is no way to
          // know in advance which productLabel or moduleLabel will be requested.
          // Maybe none will be. requestedModuleLabel and moduleLabelMismatch
          // are meaningless for "may consumes" cases.

          auto const nPreMayConsumes = resultForTransition.size();
          for (auto const& products : itRec->second) {
            if (products.first.type() == esConsumesInfo.dataKey_.type()) {
              // This is a "may consume" case, we need to find all components that may produce this dataKey
              auto const& componentDescription = products.second.componentDescription_;
              if (componentDescription) {
                info.productLabel_ = products.first.name().value();
                info.moduleType_ = componentDescription->type_;
                info.moduleLabel_ = labelForComponentDescription(componentDescription);
                info.mayConsumesNoProducts_ = false;

                info.produceMethodIDOfProducer_ = products.second.produceMethodID_;
                info.isSource_ = componentDescription->isSource_;
                info.isLooper_ = componentDescription->isLooper_;
                resultForTransition.push_back(info);
                info.mayConsumesFirstEntry_ = false;
              }
            }
          }
          if (resultForTransition.size() == nPreMayConsumes) {
            // No products can be consumed, so we add an empty entry
            // to indicate that this is a "may consumes" case with no products
            info.productLabel_ = {};
            info.mayConsumesNoProducts_ = true;
            resultForTransition.push_back(info);
          }
          // Handle cases not involving "may consumes"
        } else {
          //look for matches
          info.productLabel_ = esConsumesInfo.dataKey_.name().value();
          info.requestedModuleLabel_ = esConsumesInfo.componentLabel_;
          auto itRec = producedByESModule.find(esConsumesInfo.recordForDataKey_);
          if (itRec != producedByESModule.end()) {
            auto itProduceInfo = itRec->second.find(esConsumesInfo.dataKey_);
            if (itProduceInfo != itRec->second.end()) {
              auto const componentDescription = itProduceInfo->second.componentDescription_;
              info.moduleLabelMismatch_ =
                  ((componentDescription) and (not esConsumesInfo.componentLabel_.empty()) and
                   esConsumesInfo.componentLabel_ != labelForComponentDescription(componentDescription));
              info.mayConsumes_ = false;
              info.mayConsumesFirstEntry_ = false;
              info.mayConsumesNoProducts_ = false;

              if (componentDescription) {
                info.moduleType_ = componentDescription->type_;
                info.moduleLabel_ = labelForComponentDescription(componentDescription);
                info.produceMethodIDOfProducer_ = itProduceInfo->second.produceMethodID_;
                info.isSource_ = componentDescription->isSource_;
                info.isLooper_ = componentDescription->isLooper_;
              }
            }
          }
          resultForTransition.push_back(info);
        }
      }
      return result;
    }

  }  // namespace
  std::vector<std::vector<ESModuleConsumesInfo>> PathsAndConsumesOfModules::doESModuleConsumesInfos(
      unsigned int esModuleID) const {
    checkEventSetupInitialization();
    eventsetup::ESProductResolverProvider const* provider =
        allESProductResolverProviders_.at(esModuleIndex(esModuleID));
    ESProducer const* esProducer = dynamic_cast<ESProducer const*>(provider);
    if (esProducer) {
      return esModuleConsumesInfosCreate(*esProducer, producedByESModule_);
    }
    return {};
  }

  unsigned int PathsAndConsumesOfModules::moduleIndex(unsigned int moduleID) const {
    unsigned int dummy = 0;
    auto target = std::make_pair(moduleID, dummy);
    std::vector<std::pair<unsigned int, unsigned int>>::const_iterator iter =
        std::lower_bound(moduleIDToIndex_.begin(), moduleIDToIndex_.end(), target);
    if (iter == moduleIDToIndex_.end() || iter->first != moduleID) {
      throw Exception(errors::LogicError)
          << "PathsAndConsumesOfModules::moduleIndex: Unknown moduleID " << moduleID << "\n";
    }
    return iter->second;
  }

  unsigned int PathsAndConsumesOfModules::esModuleIndex(unsigned int esModuleID) const {
    unsigned int dummy = 0;
    auto target = std::make_pair(esModuleID, dummy);
    std::vector<std::pair<unsigned int, unsigned int>>::const_iterator iter =
        std::lower_bound(esModuleIDToIndex_.begin(), esModuleIDToIndex_.end(), target);
    if (iter == esModuleIDToIndex_.end() || iter->first != esModuleID) {
      throw Exception(errors::LogicError)
          << "PathsAndConsumesOfModules::moduleIndex: Unknown esModuleID " << esModuleID << "\n";
    }
    return iter->second;
  }
}  // namespace edm

namespace {
  // helper function for nonConsumedUnscheduledModules,
  void findAllConsumedModules(edm::PathsAndConsumesOfModulesBase const& iPnC,
                              edm::ModuleDescription const* module,
                              std::unordered_set<unsigned int>& consumedModules) {
    // If this node of the DAG has been processed already, no need to
    // reprocess again
    if (consumedModules.find(module->id()) != consumedModules.end()) {
      return;
    }
    consumedModules.insert(module->id());
    for (auto iBranchType = 0U; iBranchType != edm::NumBranchTypes; ++iBranchType) {
      for (auto const& c :
           iPnC.modulesWhoseProductsAreConsumedBy(module->id(), static_cast<edm::BranchType>(iBranchType))) {
        findAllConsumedModules(iPnC, c, consumedModules);
      }
    }
  }
}  // namespace

namespace edm {
  std::vector<ModuleDescription const*> nonConsumedUnscheduledModules(edm::PathsAndConsumesOfModulesBase const& iPnC) {
    const std::string kTriggerResults("TriggerResults");

    std::vector<std::string> pathNames = iPnC.paths();
    const unsigned int kFirstEndPathIndex = pathNames.size();
    pathNames.insert(pathNames.end(), iPnC.endPaths().begin(), iPnC.endPaths().end());

    // The goal is to find modules that are not depended upon by
    // scheduled modules. To do that, we identify all modules that are
    // depended upon by scheduled modules, and do a set subtraction.
    //
    // First, denote all scheduled modules (i.e. in Paths and
    // EndPaths) as "consumers".
    std::vector<ModuleDescription const*> consumerModules;
    for (unsigned int pathIndex = 0; pathIndex != pathNames.size(); ++pathIndex) {
      std::vector<ModuleDescription const*> const* moduleDescriptions;
      if (pathIndex < kFirstEndPathIndex) {
        moduleDescriptions = &(iPnC.modulesOnPath(pathIndex));
      } else {
        moduleDescriptions = &(iPnC.modulesOnEndPath(pathIndex - kFirstEndPathIndex));
      }
      std::copy(moduleDescriptions->begin(), moduleDescriptions->end(), std::back_inserter(consumerModules));
    }

    // Then add TriggerResults, and all Paths and EndPaths themselves
    // to the set of "consumers" (even if they don't depend on any
    // data products, they must not be deleted). Also add anything
    // consumed by child SubProcesses to the set of "consumers".
    auto const& allModules = iPnC.allModules();
    for (auto const& description : allModules) {
      if (description->moduleLabel() == kTriggerResults or
          std::find(pathNames.begin(), pathNames.end(), description->moduleLabel()) != pathNames.end()) {
        consumerModules.push_back(description);
      }
    }

    // Find modules that have any data dependence path to any module
    // in consumerModules.
    std::unordered_set<unsigned int> consumedModules;
    for (auto& description : consumerModules) {
      findAllConsumedModules(iPnC, description, consumedModules);
    }

    // All other modules will then be classified as non-consumed, even
    // if they would have dependencies within them.
    std::vector<ModuleDescription const*> unusedModules;
    std::copy_if(allModules.begin(),
                 allModules.end(),
                 std::back_inserter(unusedModules),
                 [&consumedModules](ModuleDescription const* description) {
                   return consumedModules.find(description->id()) == consumedModules.end();
                 });
    return unusedModules;
  }

  //====================================
  // checkForCorrectness algorithm
  //
  // The code creates a 'dependency' graph between all
  // modules. A module depends on another module if
  // 1) it 'consumes' data produced by that module
  // 2) it appears directly after the module within a Path
  //
  // If there is a cycle in the 'dependency' graph then
  // the schedule may be unrunnable. The schedule is still
  // runnable if all cycles have at least two edges which
  // connect modules only by Path dependencies (i.e. not
  // linked by a data dependency).
  //
  //  Example 1:
  //  C consumes data from B
  //  Path 1: A + B + C
  //  Path 2: B + C + A
  //
  //  Cycle: A after C [p2], C consumes B, B after A [p1]
  //  Since this cycle has 2 path only edges it is OK since
  //  A and (B+C) are independent so their run order doesn't matter
  //
  //  Example 2:
  //  B consumes A
  //  C consumes B
  //  Path: C + A
  //
  //  Cycle: A after C [p], C consumes B, B consumes A
  //  Since this cycle has 1 path only edge it is unrunnable.
  //
  //  Example 3:
  //  A consumes B
  //  B consumes C
  //  C consumes A
  //  (no Path since unscheduled execution)
  //
  //  Cycle: A consumes B, B consumes C, C consumes A
  //  Since this cycle has 0 path only edges it is unrunnable.
  //====================================

  namespace {
    struct ModuleStatus {
      std::vector<unsigned int> dependsOn_;
      std::vector<unsigned int> pathsOn_;
      unsigned long long lastSearch = 0;
      bool onPath_ = false;
      bool wasRun_ = false;
    };

    struct PathStatus {
      std::vector<unsigned int> modulesOnPath_;
      unsigned long int activeModuleSlot_ = 0;
      unsigned long int nModules_ = 0;
      unsigned int index_ = 0;
      bool endPath_ = false;
    };

    class CircularDependencyException {};

    bool checkIfCanRun(unsigned long long searchIndex,
                       unsigned int iModuleToCheckID,
                       std::vector<ModuleStatus>& iModules,
                       std::vector<unsigned int>& stackTrace) {
      auto& status = iModules[iModuleToCheckID];
      if (status.wasRun_) {
        return true;
      }

      if (status.lastSearch == searchIndex) {
        //check to see if the module is already on the stack
        // checking searchIndex is insufficient as multiple modules
        // in this search may be dependent upon the same module
        auto itFound = std::find(stackTrace.begin(), stackTrace.end(), iModuleToCheckID);
        if (itFound != stackTrace.end()) {
          stackTrace.push_back(iModuleToCheckID);
          throw CircularDependencyException();
        }
        //we have already checked this module's dependencies during this search
        return false;
      }
      stackTrace.push_back(iModuleToCheckID);
      status.lastSearch = searchIndex;

      bool allDependenciesRan = true;
      for (auto index : status.dependsOn_) {
        auto& dep = iModules[index];
        if (dep.onPath_) {
          if (not dep.wasRun_) {
            allDependenciesRan = false;
          }
        } else if (not checkIfCanRun(searchIndex, index, iModules, stackTrace)) {
          allDependenciesRan = false;
        }
      }
      if (allDependenciesRan) {
        status.wasRun_ = true;
      }
      stackTrace.pop_back();

      return allDependenciesRan;
    }

    void findAllDependenciesForModule(unsigned int iModID,
                                      std::vector<ModuleStatus> const& iStatus,
                                      std::vector<std::unordered_set<unsigned int>>& oDependencies) {
      auto const& dependsOn = iStatus[iModID].dependsOn_;
      if (dependsOn.empty() or !oDependencies[iModID].empty()) {
        return;
      }
      oDependencies[iModID].insert(dependsOn.begin(), dependsOn.end());
      for (auto dep : dependsOn) {
        findAllDependenciesForModule(dep, iStatus, oDependencies);
        oDependencies[iModID].merge(oDependencies[dep]);
      }
    }
    std::vector<std::unordered_set<unsigned int>> findAllDependenciesForModules(
        std::vector<ModuleStatus> const& iStatus) {
      std::vector<std::unordered_set<unsigned int>> ret(iStatus.size());
      for (unsigned int id = 0; id < iStatus.size(); ++id) {
        findAllDependenciesForModule(id, iStatus, ret);
      }
      return ret;
    }
  }  // namespace
  void checkForModuleDependencyCorrectness(edm::PathsAndConsumesOfModulesBase const& iPnC, bool iPrintDependencies) {
    constexpr auto kInvalidIndex = std::numeric_limits<unsigned int>::max();

    //Need to lookup ids to names quickly
    std::unordered_map<unsigned int, std::string> moduleIndexToNames;

    std::unordered_map<std::string, unsigned int> pathStatusInserterModuleLabelToModuleID;

    //for testing, state that TriggerResults is at the end of all paths
    const std::string kTriggerResults("TriggerResults");
    const std::string kPathStatusInserter("PathStatusInserter");
    const std::string kEndPathStatusInserter("EndPathStatusInserter");
    unsigned int kTriggerResultsIndex = kInvalidIndex;
    ModuleStatus triggerResultsStatus;
    unsigned int largestIndex = 0;
    for (auto const& description : iPnC.allModules()) {
      moduleIndexToNames.insert(std::make_pair(description->id(), description->moduleLabel()));
      if (kTriggerResults == description->moduleLabel()) {
        kTriggerResultsIndex = description->id();
      }
      if (description->id() > largestIndex) {
        largestIndex = description->id();
      }
      if (description->moduleName() == kPathStatusInserter) {
        triggerResultsStatus.dependsOn_.push_back(description->id());
      }
      if (description->moduleName() == kPathStatusInserter || description->moduleName() == kEndPathStatusInserter) {
        pathStatusInserterModuleLabelToModuleID[description->moduleLabel()] = description->id();
      }
    }

    std::vector<ModuleStatus> statusOfModules(largestIndex + 1);
    for (auto const& nameID : pathStatusInserterModuleLabelToModuleID) {
      statusOfModules[nameID.second].onPath_ = true;
      unsigned int pathIndex;
      auto const& paths = iPnC.paths();
      auto itFound = std::find(paths.begin(), paths.end(), nameID.first);
      if (itFound != paths.end()) {
        pathIndex = itFound - paths.begin();
      } else {
        auto const& endPaths = iPnC.endPaths();
        itFound = std::find(endPaths.begin(), endPaths.end(), nameID.first);
        assert(itFound != endPaths.end());
        pathIndex = itFound - endPaths.begin() + iPnC.paths().size();
      }
      statusOfModules[nameID.second].pathsOn_.push_back(pathIndex);
    }
    if (kTriggerResultsIndex != kInvalidIndex) {
      statusOfModules[kTriggerResultsIndex] = std::move(triggerResultsStatus);
    }

    std::vector<PathStatus> statusOfPaths(iPnC.paths().size() + iPnC.endPaths().size());

    //If there are no paths, no modules will run so nothing to check
    if (statusOfPaths.empty()) {
      return;
    }

    {
      auto nPaths = iPnC.paths().size();
      for (unsigned int p = 0; p < nPaths; ++p) {
        auto& status = statusOfPaths[p];
        status.index_ = p;
        status.modulesOnPath_.reserve(iPnC.modulesOnPath(p).size() + 1);
        std::unordered_set<unsigned int> uniqueModules;
        for (auto const& mod : iPnC.modulesOnPath(p)) {
          if (uniqueModules.insert(mod->id()).second) {
            status.modulesOnPath_.push_back(mod->id());
            statusOfModules[mod->id()].onPath_ = true;
            statusOfModules[mod->id()].pathsOn_.push_back(p);
          }
        }
        status.nModules_ = uniqueModules.size() + 1;

        //add the PathStatusInserter at the end
        auto found = pathStatusInserterModuleLabelToModuleID.find(iPnC.paths()[p]);
        assert(found != pathStatusInserterModuleLabelToModuleID.end());
        status.modulesOnPath_.push_back(found->second);
      }
    }
    {
      auto offset = iPnC.paths().size();
      auto nPaths = iPnC.endPaths().size();
      for (unsigned int p = 0; p < nPaths; ++p) {
        auto& status = statusOfPaths[p + offset];
        status.endPath_ = true;
        status.index_ = p;
        status.modulesOnPath_.reserve(iPnC.modulesOnEndPath(p).size() + 1);
        std::unordered_set<unsigned int> uniqueModules;
        for (auto const& mod : iPnC.modulesOnEndPath(p)) {
          if (uniqueModules.insert(mod->id()).second) {
            status.modulesOnPath_.push_back(mod->id());
            statusOfModules[mod->id()].onPath_ = true;
            statusOfModules[mod->id()].pathsOn_.push_back(p + offset);
          }
        }
        status.nModules_ = uniqueModules.size();

        //add the EndPathStatusInserter at the end
        auto found = pathStatusInserterModuleLabelToModuleID.find(iPnC.endPaths()[p]);
        if (found != pathStatusInserterModuleLabelToModuleID.end()) {
          status.modulesOnPath_.push_back(found->second);
          ++status.nModules_;
        }
      }
    }

    for (auto const& description : iPnC.allModules()) {
      unsigned int const moduleIndex = description->id();
      auto const& dependentModules = iPnC.modulesWhoseProductsAreConsumedBy(moduleIndex);
      auto& deps = statusOfModules[moduleIndex];
      deps.dependsOn_.reserve(dependentModules.size());
      for (auto const& depDescription : dependentModules) {
        if (iPrintDependencies) {
          edm::LogAbsolute("ModuleDependency")
              << "ModuleDependency '" << description->moduleLabel() << "' depends on data products from module '"
              << depDescription->moduleLabel() << "'";
        }
        deps.dependsOn_.push_back(depDescription->id());
      }
    }

    unsigned int nPathsFinished = 0;
    for (auto const& status : statusOfPaths) {
      if (status.nModules_ == 0) {
        ++nPathsFinished;
      }
    }

    //if a circular dependency exception happens, stackTrace has the info
    std::vector<unsigned int> stackTrace;
    bool madeForwardProgress = true;
    try {
      //'simulate' the running of the paths. On each step mark each module as 'run'
      // if all the module's dependencies were fulfilled in a previous step
      unsigned long long searchIndex = 0;
      while (madeForwardProgress and nPathsFinished != statusOfPaths.size()) {
        madeForwardProgress = false;
        for (auto& p : statusOfPaths) {
          //the path has already completed in an earlier pass
          if (p.activeModuleSlot_ == p.nModules_) {
            continue;
          }
          ++searchIndex;
          bool didRun = checkIfCanRun(searchIndex, p.modulesOnPath_[p.activeModuleSlot_], statusOfModules, stackTrace);
          if (didRun) {
            madeForwardProgress = true;
            ++p.activeModuleSlot_;
            if (p.activeModuleSlot_ == p.nModules_) {
              ++nPathsFinished;
            }
          }
        }
      }
    } catch (CircularDependencyException const&) {
      //the last element in stackTrace must appear somewhere earlier in stackTrace
      std::ostringstream oStr;

      unsigned int lastIndex = stackTrace.front();
      bool firstSkipped = false;
      for (auto id : stackTrace) {
        if (firstSkipped) {
          oStr << "  module '" << moduleIndexToNames[lastIndex] << "' depends on " << moduleIndexToNames[id] << "\n";
        } else {
          firstSkipped = true;
        }
        lastIndex = id;
      }
      throw edm::Exception(edm::errors::ScheduleExecutionFailure, "Unrunnable schedule\n")
          << "Circular module dependency found in configuration\n"
          << oStr.str();
    }

    auto pathName = [&](PathStatus const& iP) {
      if (iP.endPath_) {
        return iPnC.endPaths()[iP.index_];
      }
      return iPnC.paths()[iP.index_];
    };

    //The program would deadlock
    if (not madeForwardProgress) {
      std::ostringstream oStr;
      auto modIndex = std::numeric_limits<unsigned int>::max();
      unsigned int presentPath;
      for (auto itP = statusOfPaths.begin(); itP != statusOfPaths.end(); ++itP) {
        auto const& p = *itP;
        if (p.activeModuleSlot_ == p.nModules_) {
          continue;
        }
        //this path is stuck
        modIndex = p.modulesOnPath_[p.activeModuleSlot_];
        presentPath = itP - statusOfPaths.begin();
        break;
      }
      //NOTE the following should always be true as at least 1 path should be stuc.
      // I've added the condition just to be paranoid.
      if (modIndex != std::numeric_limits<unsigned int>::max()) {
        struct ProgressInfo {
          ProgressInfo(unsigned int iMod, unsigned int iPath, bool iPreceeds = false)
              : moduleIndex_(iMod), pathIndex_(iPath), preceeds_(iPreceeds) {}

          ProgressInfo(unsigned int iMod) : moduleIndex_(iMod), pathIndex_{}, preceeds_(false) {}

          unsigned int moduleIndex_ = std::numeric_limits<unsigned int>::max();
          std::optional<unsigned int> pathIndex_;
          bool preceeds_;

          bool operator==(ProgressInfo const& iOther) const {
            return moduleIndex_ == iOther.moduleIndex_ and pathIndex_ == iOther.pathIndex_;
          }
        };

        std::vector<ProgressInfo> progressTrace;
        progressTrace.emplace_back(modIndex, presentPath);

        //The following starts from the first found unrun module on a path. It then finds
        // the first modules it depends on that was not run. If that module is on a Task
        // it then repeats the check for that module's dependencies. If that module is on
        // a path, it checks to see if that module is the first unrun module of a path
        // and if so it repeats the check for that module's dependencies, if not it
        // checks the dependencies of the stuck module on that path.
        // Eventually, all these checks should allow us to find a cycle of modules.

        //NOTE: the only way foundUnrunModule should ever by false by the end of the
        // do{}while loop is if there is a bug in the algorithm. I've included it to
        // try to avoid that case causing an infinite loop in the program.
        bool foundUnrunModule;
        do {
          //check dependencies looking for stuff not run and on a path
          foundUnrunModule = false;
          for (auto depMod : statusOfModules[modIndex].dependsOn_) {
            auto const& depStatus = statusOfModules[depMod];
            if (not depStatus.wasRun_ and depStatus.onPath_) {
              foundUnrunModule = true;
              //last run on a path?
              bool lastOnPath = false;
              unsigned int foundPath;
              for (auto pathOn : depStatus.pathsOn_) {
                auto const& depPaths = statusOfPaths[pathOn];
                if (depPaths.modulesOnPath_[depPaths.activeModuleSlot_] == depMod) {
                  lastOnPath = true;
                  foundPath = pathOn;
                  break;
                }
              }
              if (lastOnPath) {
                modIndex = depMod;
                progressTrace.emplace_back(modIndex, foundPath);
              } else {
                //some earlier module on the same path is stuck
                progressTrace.emplace_back(depMod, depStatus.pathsOn_[0]);
                auto const& depPath = statusOfPaths[depStatus.pathsOn_[0]];
                modIndex = depPath.modulesOnPath_[depPath.activeModuleSlot_];
                progressTrace.emplace_back(modIndex, depStatus.pathsOn_[0], true);
              }
              break;
            }
          }
          if (not foundUnrunModule) {
            //check unscheduled modules
            for (auto depMod : statusOfModules[modIndex].dependsOn_) {
              auto const& depStatus = statusOfModules[depMod];
              if (not depStatus.wasRun_ and not depStatus.onPath_) {
                foundUnrunModule = true;
                progressTrace.emplace_back(depMod);
                modIndex = depMod;
                break;
              }
            }
          }
        } while (foundUnrunModule and (0 == std::count(progressTrace.begin(),
                                                       progressTrace.begin() + progressTrace.size() - 1,
                                                       progressTrace.back())));

        auto printTrace = [&](auto& oStr, auto itBegin, auto itEnd) {
          for (auto itTrace = itBegin; itTrace != itEnd; ++itTrace) {
            if (itTrace != itBegin) {
              if (itTrace->preceeds_) {
                oStr << " and follows module '" << moduleIndexToNames[itTrace->moduleIndex_] << "' on the path\n";
              } else {
                oStr << " and depends on module '" << moduleIndexToNames[itTrace->moduleIndex_] << "'\n";
              }
            }
            if (itTrace + 1 != itEnd) {
              if (itTrace->pathIndex_) {
                oStr << "  module '" << moduleIndexToNames[itTrace->moduleIndex_] << "' is on path '"
                     << pathName(statusOfPaths[*itTrace->pathIndex_]) << "'";
              } else {
                oStr << "  module '" << moduleIndexToNames[itTrace->moduleIndex_] << "' is in a task";
              }
            }
          }
        };

        if (not foundUnrunModule) {
          //If we get here, this suggests a problem with either the algorithm that finds problems or the algorithm
          // that attempts to report the problem
          oStr << "Algorithm Error, unable to find problem. Contact framework group.\n Traced problem this far\n";
          printTrace(oStr, progressTrace.begin(), progressTrace.end());
        } else {
          printTrace(
              oStr, std::find(progressTrace.begin(), progressTrace.end(), progressTrace.back()), progressTrace.end());
        }
      }
      //the schedule deadlocked
      throw edm::Exception(edm::errors::ScheduleExecutionFailure, "Unrunnable schedule\n")
          << "The Path/EndPath configuration could cause the job to deadlock\n"
          << oStr.str();
    }

    //NOTE: although the following conditions are not needed for safe running, they are
    // policy choices the collaboration has made.

    //HLT wants all paths to be equivalent. If a path has a module A that needs data from module B and module B appears on one path
    // as module A then B must appear on ALL paths that have A.
    unsigned int modIndex = 0;
    for (auto& mod : statusOfModules) {
      for (auto& depIndex : mod.dependsOn_) {
        std::size_t count = 0;
        std::size_t nonEndPaths = 0;
        for (auto modPathID : mod.pathsOn_) {
          if (statusOfPaths[modPathID].endPath_) {
            continue;
          }
          ++nonEndPaths;
          for (auto depPathID : statusOfModules[depIndex].pathsOn_) {
            if (depPathID == modPathID) {
              ++count;
              break;
            }
          }
        }
        if (count != 0 and count != nonEndPaths) {
          std::ostringstream onStr;
          std::ostringstream missingStr;

          for (auto modPathID : mod.pathsOn_) {
            if (statusOfPaths[modPathID].endPath_) {
              continue;
            }
            bool found = false;
            for (auto depPathID : statusOfModules[depIndex].pathsOn_) {
              if (depPathID == modPathID) {
                found = true;
              }
            }
            auto& s = statusOfPaths[modPathID];
            if (found) {
              onStr << pathName(s) << " ";
            } else {
              missingStr << pathName(s) << " ";
            }
          }
          throw edm::Exception(edm::errors::ScheduleExecutionFailure, "Unrunnable schedule\n")
              << "Paths are non consistent\n"
              << "  module '" << moduleIndexToNames[modIndex] << "' depends on '" << moduleIndexToNames[depIndex]
              << "' which appears on paths\n  " << onStr.str() << "\nbut is missing from\n  " << missingStr.str();
        }
      }
      ++modIndex;
    }

    //Check to see if for each path if the order of the modules is correct based on dependencies
    auto allDependencies = findAllDependenciesForModules(statusOfModules);
    for (auto& p : statusOfPaths) {
      for (unsigned long int i = 0; p.nModules_ > 0 and i < p.nModules_ - 1; ++i) {
        auto moduleID = p.modulesOnPath_[i];
        if (not allDependencies[moduleID].empty()) {
          for (unsigned long int j = i + 1; j < p.nModules_; ++j) {
            auto testModuleID = p.modulesOnPath_[j];
            if (allDependencies[moduleID].find(testModuleID) != allDependencies[moduleID].end()) {
              throw edm::Exception(edm::errors::ScheduleExecutionFailure, "Unrunnable schedule\n")
                  << "Dependent module later on Path\n"
                  << "  module '" << moduleIndexToNames[moduleID] << "' depends on '"
                  << moduleIndexToNames[testModuleID] << "' which is later on path " << pathName(p);
            }
          }
        }
      }
    }
  }
}  // namespace edm
