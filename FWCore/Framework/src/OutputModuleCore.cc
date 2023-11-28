// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     OutputModuleCore
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Wed, 31 Jul 2013 15:59:19 GMT
//

// system include files
#include <cassert>

// user include files
#include "FWCore/Framework/interface/OutputModuleCore.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Common/interface/EndPathStatus.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/src/insertSelectedProcesses.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/ProcessBlockForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/src/OutputModuleDescription.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Reflection/interface/DictionaryTools.h"

namespace edm {
  namespace core {

    // -------------------------------------------------------
    OutputModuleCore::OutputModuleCore(ParameterSet const& pset)
        : remainingEvents_(-1),
          maxEvents_(-1),
          keptProducts_(),
          hasNewlyDroppedBranch_(),
          process_name_(),
          productSelectorRules_(pset, "outputCommands", "OutputModule"),
          productSelector_(),
          moduleDescription_(),
          wantAllEvents_(false),
          selectors_(),
          selector_config_id_(),
          droppedBranchIDToKeptBranchID_(),
          branchIDLists_(new BranchIDLists),
          origBranchIDLists_(nullptr),
          thinnedAssociationsHelper_(new ThinnedAssociationsHelper) {
      hasNewlyDroppedBranch_.fill(false);

      Service<service::TriggerNamesService> tns;
      process_name_ = tns->getProcessName();

      selectEvents_ = pset.getUntrackedParameterSet("SelectEvents", ParameterSet());

      selectEvents_.registerIt();  // Just in case this PSet is not registered

      selector_config_id_ = selectEvents_.id();

      //need to set wantAllEvents_ in constructor
      // we will make the remaining selectors once we know how many streams
      selectors_.resize(1);
      wantAllEvents_ = detail::configureEventSelector(
          selectEvents_, process_name_, getAllTriggerNames(), selectors_[0], consumesCollector());

      //Check if on final path
      if (pset.exists("@onFinalPath")) {
        onFinalPath_ = pset.getUntrackedParameter<bool>("@onFinalPath");
      }
      if (onFinalPath_) {
        wantAllEvents_ = false;
        if (not getAllTriggerNames().empty() and selectors_.front().numberOfTokens() == 0) {
          //need to wait for trigger paths to finish
          tokensForEndPaths_.push_back(consumes<TriggerResults>(edm::InputTag("TriggerResults", "", process_name_)));
        }
        //need to wait for EndPaths to finish
        for (auto const& n : tns->getEndPaths()) {
          if (n == "@finalPath") {
            continue;
          }
          tokensForEndPaths_.push_back(consumes<EndPathStatus>(edm::InputTag(n, "", process_name_)));
        }
      }
    }

    void OutputModuleCore::configure(OutputModuleDescription const& desc) {
      remainingEvents_ = maxEvents_ = desc.maxEvents_;
      origBranchIDLists_ = desc.branchIDLists_;
    }

    void OutputModuleCore::selectProducts(ProductRegistry const& preg,
                                          ThinnedAssociationsHelper const& thinnedAssociationsHelper,
                                          ProcessBlockHelperBase const& processBlockHelper) {
      if (productSelector_.initialized())
        return;
      productSelector_.initialize(productSelectorRules_, preg.allBranchDescriptions());

      // TODO: See if we can collapse keptProducts_ and productSelector_ into a
      // single object. See the notes in the header for ProductSelector
      // for more information.

      std::map<BranchID, BranchDescription const*> trueBranchIDToKeptBranchDesc;
      std::vector<BranchDescription const*> associationDescriptions;
      std::set<BranchID> keptProductsInEvent;
      std::set<std::string> processesWithSelectedMergeableRunProducts;
      std::set<std::string> processesWithKeptProcessBlockProducts;

      for (auto const& it : preg.productList()) {
        BranchDescription const& desc = it.second;
        if (desc.transient()) {
          // if the class of the branch is marked transient, output nothing
        } else if (!desc.present() && !desc.produced()) {
          // else if the branch containing the product has been previously dropped,
          // output nothing
        } else if (desc.unwrappedType() == typeid(ThinnedAssociation)) {
          associationDescriptions.push_back(&desc);
        } else if (selected(desc)) {
          keepThisBranch(desc, trueBranchIDToKeptBranchDesc, keptProductsInEvent);
          insertSelectedProcesses(
              desc, processesWithSelectedMergeableRunProducts, processesWithKeptProcessBlockProducts);
        } else {
          // otherwise, output nothing,
          // and mark the fact that there is a newly dropped branch of this type.
          hasNewlyDroppedBranch_[desc.branchType()] = true;
        }
      }

      setProcessesWithSelectedMergeableRunProducts(processesWithSelectedMergeableRunProducts);

      thinnedAssociationsHelper.selectAssociationProducts(
          associationDescriptions, keptProductsInEvent, keepAssociation_);

      for (auto association : associationDescriptions) {
        if (keepAssociation_[association->branchID()]) {
          keepThisBranch(*association, trueBranchIDToKeptBranchDesc, keptProductsInEvent);
        } else {
          hasNewlyDroppedBranch_[association->branchType()] = true;
        }
      }

      // Now fill in a mapping needed in the case that a branch was dropped while its EDAlias was kept.
      ProductSelector::fillDroppedToKept(preg, trueBranchIDToKeptBranchDesc, droppedBranchIDToKeptBranchID_);

      thinnedAssociationsHelper_->updateFromParentProcess(
          thinnedAssociationsHelper, keepAssociation_, droppedBranchIDToKeptBranchID_);
      outputProcessBlockHelper_.updateAfterProductSelection(processesWithKeptProcessBlockProducts, processBlockHelper);
    }

    void OutputModuleCore::updateBranchIDListsWithKeptAliases() {
      if (!droppedBranchIDToKeptBranchID_.empty()) {
        // Make a private copy of the BranchIDLists.
        *branchIDLists_ = *origBranchIDLists_;
        // Check for branches dropped while an EDAlias was kept.
        for (BranchIDList& branchIDList : *branchIDLists_) {
          for (BranchID::value_type& branchID : branchIDList) {
            // Replace BranchID of each dropped branch with that of the kept
            // alias, so the alias branch will have the product ID of the original branch.
            std::map<BranchID::value_type, BranchID::value_type>::const_iterator iter =
                droppedBranchIDToKeptBranchID_.find(branchID);
            if (iter != droppedBranchIDToKeptBranchID_.end()) {
              branchID = iter->second;
            }
          }
        }
      }
    }

    void OutputModuleCore::keepThisBranch(BranchDescription const& desc,
                                          std::map<BranchID, BranchDescription const*>& trueBranchIDToKeptBranchDesc,
                                          std::set<BranchID>& keptProductsInEvent) {
      ProductSelector::checkForDuplicateKeptBranch(desc, trueBranchIDToKeptBranchDesc);

      EDGetToken token;

      std::vector<std::string> missingDictionaries;
      if (!checkDictionary(missingDictionaries, desc.className(), desc.unwrappedType())) {
        std::string context("Calling OutputModuleCore::keepThisBranch, checking dictionaries for kept types");
        throwMissingDictionariesException(missingDictionaries, context);
      }

      switch (desc.branchType()) {
        case InEvent: {
          if (desc.produced()) {
            keptProductsInEvent.insert(desc.originalBranchID());
          } else {
            keptProductsInEvent.insert(desc.branchID());
          }
          token = consumes(TypeToGet{desc.unwrappedTypeID(), PRODUCT_TYPE},
                           InputTag{desc.moduleLabel(), desc.productInstanceName(), desc.processName()});
          break;
        }
        case InLumi: {
          token = consumes<InLumi>(TypeToGet{desc.unwrappedTypeID(), PRODUCT_TYPE},
                                   InputTag(desc.moduleLabel(), desc.productInstanceName(), desc.processName()));
          break;
        }
        case InRun: {
          token = consumes<InRun>(TypeToGet{desc.unwrappedTypeID(), PRODUCT_TYPE},
                                  InputTag(desc.moduleLabel(), desc.productInstanceName(), desc.processName()));
          break;
        }
        case InProcess: {
          token = consumes<InProcess>(TypeToGet{desc.unwrappedTypeID(), PRODUCT_TYPE},
                                      InputTag(desc.moduleLabel(), desc.productInstanceName(), desc.processName()));
          break;
        }
        default:
          assert(false);
          break;
      }
      // Now put it in the list of selected branches.
      keptProducts_[desc.branchType()].push_back(std::make_pair(&desc, token));
    }

    OutputModuleCore::~OutputModuleCore() {}

    void OutputModuleCore::doPreallocate_(PreallocationConfiguration const& iPC) {
      auto nstreams = iPC.numberOfStreams();
      selectors_.resize(nstreams);

      preallocLumis(iPC.numberOfLuminosityBlocks());

      bool seenFirst = false;
      for (auto& s : selectors_) {
        if (seenFirst) {
          detail::configureEventSelector(selectEvents_, process_name_, getAllTriggerNames(), s, consumesCollector());
        } else {
          seenFirst = true;
        }
      }
    }

    void OutputModuleCore::preallocLumis(unsigned int) {}

    void OutputModuleCore::doBeginJob_() {
      this->beginJob();
      if (onFinalPath_) {
        //this stops prefetching of the data products
        resetItemsToGetFrom(edm::InEvent);
      }
    }

    void OutputModuleCore::doEndJob() { endJob(); }

    void OutputModuleCore::registerProductsAndCallbacks(OutputModuleCore const*, ProductRegistry* reg) {
      if (callWhenNewProductsRegistered_) {
        reg->callForEachBranch(callWhenNewProductsRegistered_);

        Service<ConstProductRegistry> regService;
        regService->watchProductAdditions(callWhenNewProductsRegistered_);
      }
    }

    bool OutputModuleCore::needToRunSelection() const { return !wantAllEvents_; }

    std::vector<ProductResolverIndexAndSkipBit> OutputModuleCore::productsUsedBySelection() const {
      std::vector<ProductResolverIndexAndSkipBit> returnValue;
      auto const& s = selectors_[0];
      auto const n = s.numberOfTokens();
      returnValue.reserve(n + tokensForEndPaths_.size());

      for (unsigned int i = 0; i < n; ++i) {
        returnValue.emplace_back(uncheckedIndexFrom(s.token(i)));
      }
      for (auto token : tokensForEndPaths_) {
        returnValue.emplace_back(uncheckedIndexFrom(token));
      }
      return returnValue;
    }

    bool OutputModuleCore::prePrefetchSelection(StreamID id,
                                                EventPrincipal const& ep,
                                                ModuleCallingContext const* mcc) {
      if (wantAllEvents_)
        return true;
      auto& s = selectors_[id.value()];
      EventForOutput e(ep, moduleDescription_, mcc);
      e.setConsumer(this);
      return s.wantEvent(e);
    }

    bool OutputModuleCore::doEvent_(EventTransitionInfo const& info,
                                    ActivityRegistry* act,
                                    ModuleCallingContext const* mcc) {
      {
        EventForOutput e(info, moduleDescription_, mcc);
        e.setConsumer(this);
        EventSignalsSentry sentry(act, mcc);
        write(e);
      }
      //remainingEvents_ is decremented by inheriting classes
      return true;
    }

    bool OutputModuleCore::doBeginRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      RunForOutput r(info, moduleDescription_, mcc, false);
      r.setConsumer(this);
      doBeginRun_(r);
      return true;
    }

    bool OutputModuleCore::doEndRun(RunTransitionInfo const& info, ModuleCallingContext const* mcc) {
      RunForOutput r(info, moduleDescription_, mcc, true);
      r.setConsumer(this);
      doEndRun_(r);
      return true;
    }

    void OutputModuleCore::doWriteProcessBlock(ProcessBlockPrincipal const& pbp, ModuleCallingContext const* mcc) {
      ProcessBlockForOutput pb(pbp, moduleDescription_, mcc, true);
      pb.setConsumer(this);
      writeProcessBlock(pb);
    }

    void OutputModuleCore::doWriteRun(RunPrincipal const& rp,
                                      ModuleCallingContext const* mcc,
                                      MergeableRunProductMetadata const* mrpm) {
      RunForOutput r(rp, moduleDescription_, mcc, true, mrpm);
      r.setConsumer(this);
      writeRun(r);
    }

    bool OutputModuleCore::doBeginLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) {
      LuminosityBlockForOutput lb(info, moduleDescription_, mcc, false);
      lb.setConsumer(this);
      doBeginLuminosityBlock_(lb);
      return true;
    }

    bool OutputModuleCore::doEndLuminosityBlock(LumiTransitionInfo const& info, ModuleCallingContext const* mcc) {
      LuminosityBlockForOutput lb(info, moduleDescription_, mcc, true);
      lb.setConsumer(this);
      doEndLuminosityBlock_(lb);

      return true;
    }

    void OutputModuleCore::doWriteLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                                  ModuleCallingContext const* mcc) {
      LuminosityBlockForOutput lb(lbp, moduleDescription_, mcc, true);
      lb.setConsumer(this);
      writeLuminosityBlock(lb);
    }

    void OutputModuleCore::doOpenFile(FileBlock const& fb) { openFile(fb); }

    void OutputModuleCore::doRespondToOpenInputFile(FileBlock const& fb) {
      updateBranchIDListsWithKeptAliases();
      doRespondToOpenInputFile_(fb);
    }

    void OutputModuleCore::doRespondToCloseInputFile(FileBlock const& fb) { doRespondToCloseInputFile_(fb); }

    void OutputModuleCore::doCloseFile() {
      if (isFileOpen()) {
        reallyCloseFile();
      }
    }

    void OutputModuleCore::reallyCloseFile() {}

    BranchIDLists const* OutputModuleCore::branchIDLists() const {
      if (!droppedBranchIDToKeptBranchID_.empty()) {
        return branchIDLists_.get();
      }
      return origBranchIDLists_;
    }

    ThinnedAssociationsHelper const* OutputModuleCore::thinnedAssociationsHelper() const {
      return thinnedAssociationsHelper_.get();
    }

    ModuleDescription const& OutputModuleCore::description() const { return moduleDescription_; }

    bool OutputModuleCore::selected(BranchDescription const& desc) const { return productSelector_.selected(desc); }

    void OutputModuleCore::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setUnknown();
      descriptions.addDefault(desc);
    }

    void OutputModuleCore::fillDescription(ParameterSetDescription& desc,
                                           std::vector<std::string> const& defaultOutputCommands) {
      ProductSelectorRules::fillDescription(desc, "outputCommands", defaultOutputCommands);
      EventSelector::fillDescription(desc);
      desc.addOptionalNode(ParameterDescription<bool>("@onFinalPath", false, false), false);
    }

    void OutputModuleCore::prevalidate(ConfigurationDescriptions&) {}

    static const std::string kBaseType("OutputModule");
    const std::string& OutputModuleCore::baseType() { return kBaseType; }

    void OutputModuleCore::setEventSelectionInfo(
        std::map<std::string, std::vector<std::pair<std::string, int>>> const& outputModulePathPositions,
        bool anyProductProduced) {
      selector_config_id_ = detail::registerProperSelectionInfo(getParameterSet(selector_config_id_),
                                                                description().moduleLabel(),
                                                                outputModulePathPositions,
                                                                anyProductProduced);
    }
  }  // namespace core
}  // namespace edm
