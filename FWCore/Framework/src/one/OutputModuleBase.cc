// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     OutputModuleBase
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
#include "FWCore/Framework/interface/one/OutputModuleBase.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/OutputModuleDescription.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/src/EventSignalsSentry.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/DebugMacros.h"


namespace edm {
  namespace one {

    // -------------------------------------------------------
    OutputModuleBase::OutputModuleBase(ParameterSet const& pset) :
    maxEvents_(-1),
    remainingEvents_(maxEvents_),
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
      
      selectEvents_ =
      pset.getUntrackedParameterSet("SelectEvents", ParameterSet());
      
      selectEvents_.registerIt(); // Just in case this PSet is not registered
      
      selector_config_id_ = selectEvents_.id();
      
      //need to set wantAllEvents_ in constructor
      // we will make the remaining selectors once we know how many streams
      selectors_.resize(1);
      wantAllEvents_ = detail::configureEventSelector(selectEvents_,
                                                      process_name_,
                                                      getAllTriggerNames(),
                                                      selectors_[0]);

    }
    
    void OutputModuleBase::configure(OutputModuleDescription const& desc) {
      remainingEvents_ = maxEvents_ = desc.maxEvents_;
      origBranchIDLists_ = desc.branchIDLists_;
    }
    
    void OutputModuleBase::selectProducts(ProductRegistry const& preg,
                                          ThinnedAssociationsHelper const& thinnedAssociationsHelper) {
      if(productSelector_.initialized()) return;
      productSelector_.initialize(productSelectorRules_, preg.allBranchDescriptions());
      
      // TODO: See if we can collapse keptProducts_ and productSelector_ into a
      // single object. See the notes in the header for ProductSelector
      // for more information.
      
      std::map<BranchID, BranchDescription const*> trueBranchIDToKeptBranchDesc;
      std::vector<BranchDescription const*> associationDescriptions;
      std::set<BranchID> keptProductsInEvent;

      for(auto const& it : preg.productList()) {
        BranchDescription const& desc = it.second;
        if(desc.transient()) {
          // if the class of the branch is marked transient, output nothing
        } else if(!desc.present() && !desc.produced()) {
          // else if the branch containing the product has been previously dropped,
          // output nothing
        } else if(desc.unwrappedType() == typeid(ThinnedAssociation)) {
          associationDescriptions.push_back(&desc);
        } else if(selected(desc)) {
          keepThisBranch(desc, trueBranchIDToKeptBranchDesc, keptProductsInEvent);
        } else {
          // otherwise, output nothing,
          // and mark the fact that there is a newly dropped branch of this type.
          hasNewlyDroppedBranch_[desc.branchType()] = true;
        }
      }

      thinnedAssociationsHelper.selectAssociationProducts(associationDescriptions,
                                                          keptProductsInEvent,
                                                          keepAssociation_);

      for(auto association : associationDescriptions) {
        if(keepAssociation_[association->branchID()]) {
          keepThisBranch(*association, trueBranchIDToKeptBranchDesc, keptProductsInEvent);
        } else {
          hasNewlyDroppedBranch_[association->branchType()] = true;
        }
      }

      // Now fill in a mapping needed in the case that a branch was dropped while its EDAlias was kept.
      ProductSelector::fillDroppedToKept(preg, trueBranchIDToKeptBranchDesc, droppedBranchIDToKeptBranchID_);

      thinnedAssociationsHelper_->updateFromParentProcess(thinnedAssociationsHelper, keepAssociation_, droppedBranchIDToKeptBranchID_);
    }

    void OutputModuleBase::keepThisBranch(BranchDescription const& desc,
                        std::map<BranchID, BranchDescription const*>& trueBranchIDToKeptBranchDesc,
                        std::set<BranchID>& keptProductsInEvent) {

      ProductSelector::checkForDuplicateKeptBranch(desc,
                                                   trueBranchIDToKeptBranchDesc);

      switch (desc.branchType()) {
      case InEvent:
        {
          if(desc.produced()) {
            keptProductsInEvent.insert(desc.originalBranchID());
          } else {
            keptProductsInEvent.insert(desc.branchID());
          }
          consumes(TypeToGet{desc.unwrappedTypeID(),PRODUCT_TYPE},
                   InputTag{desc.moduleLabel(),
                       desc.productInstanceName(),
                       desc.processName()});
          break;
        }
      case InLumi:
        {
          consumes<InLumi>(TypeToGet{desc.unwrappedTypeID(),PRODUCT_TYPE},
                           InputTag(desc.moduleLabel(),
                                    desc.productInstanceName(),
                                    desc.processName()));
          break;
        }
      case InRun:
        {
          consumes<InRun>(TypeToGet{desc.unwrappedTypeID(),PRODUCT_TYPE},
                          InputTag(desc.moduleLabel(),
                                   desc.productInstanceName(),
                                   desc.processName()));
          break;
        }
      default:
        assert(false);
        break;
      }
      // Now put it in the list of selected branches.
      keptProducts_[desc.branchType()].push_back(&desc);
    }

    OutputModuleBase::~OutputModuleBase() { }
    
    SharedResourcesAcquirer OutputModuleBase::createAcquirer() {
      return SharedResourcesAcquirer{};
    }
    
    void OutputModuleBase::doPreallocate(PreallocationConfiguration const& iPC) {
      auto nstreams = iPC.numberOfStreams();
      selectors_.resize(nstreams);
      
      bool seenFirst = false;
      for(auto& s : selectors_) {
        if(seenFirst) {
          detail::configureEventSelector(selectEvents_,
                                         process_name_,
                                         getAllTriggerNames(),
                                         s);
        } else {
          seenFirst = true;
        }
      }
    }

    void OutputModuleBase::doBeginJob() {
      resourcesAcquirer_ = createAcquirer();
      this->beginJob();
    }
    
    void OutputModuleBase::doEndJob() {
      endJob();
    }
    
    bool OutputModuleBase::prePrefetchSelection(StreamID id, EventPrincipal const& ep, ModuleCallingContext const* mcc) {
      
      auto& s = selectors_[id.value()];
      return wantAllEvents_ or s.wantEvent(ep,mcc);
    }
    
    bool
    OutputModuleBase::doEvent(EventPrincipal const& ep,
                              EventSetup const&,
                              ActivityRegistry* act,
                              ModuleCallingContext const* mcc) {
      
      {
        std::lock_guard<std::mutex> guard(mutex_);
        {
          std::lock_guard<SharedResourcesAcquirer> guard(resourcesAcquirer_);
          EventSignalsSentry sentry(act,mcc);
          write(ep, mcc);
        }
      }
      if(remainingEvents_ > 0) {
        --remainingEvents_;
      }
      return true;
    }
    
    bool
    OutputModuleBase::doBeginRun(RunPrincipal const& rp,
                                 EventSetup const&,
                                 ModuleCallingContext const* mcc) {
      doBeginRun_(rp, mcc);
      return true;
    }
    
    bool
    OutputModuleBase::doEndRun(RunPrincipal const& rp,
                               EventSetup const&,
                               ModuleCallingContext const* mcc) {
      doEndRun_(rp, mcc);
      return true;
    }
    
    void
    OutputModuleBase::doWriteRun(RunPrincipal const& rp,
                                 ModuleCallingContext const* mcc) {
      writeRun(rp, mcc);
    }
    
    bool
    OutputModuleBase::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                             EventSetup const&,
                                             ModuleCallingContext const* mcc) {
      doBeginLuminosityBlock_(lbp, mcc);
      return true;
    }
    
    bool
    OutputModuleBase::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                           EventSetup const&,
                                           ModuleCallingContext const* mcc) {
      doEndLuminosityBlock_(lbp, mcc);
      return true;
    }
    
    void OutputModuleBase::doWriteLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                                  ModuleCallingContext const* mcc) {
      writeLuminosityBlock(lbp, mcc);
    }
    
    void OutputModuleBase::doOpenFile(FileBlock const& fb) {
      openFile(fb);
    }
    
    void OutputModuleBase::doRespondToOpenInputFile(FileBlock const& fb) {
      doRespondToOpenInputFile_(fb);
    }
    
    void OutputModuleBase::doRespondToCloseInputFile(FileBlock const& fb) {
      doRespondToCloseInputFile_(fb);
    }
    
    void
    OutputModuleBase::doPreForkReleaseResources() {
      preForkReleaseResources();
    }
    
    void
    OutputModuleBase::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
      postForkReacquireResources(iChildIndex, iNumberOfChildren);
    }
    
    void
    OutputModuleBase::preForkReleaseResources() {}
    
    void
    OutputModuleBase::postForkReacquireResources(unsigned int /*iChildIndex*/, unsigned int /*iNumberOfChildren*/) {}

    
    void OutputModuleBase::maybeOpenFile() {
      if(!isFileOpen()) reallyOpenFile();
    }
    
    void OutputModuleBase::doCloseFile() {
      if(isFileOpen()) {
        reallyCloseFile();
      }
    }
    
    void OutputModuleBase::reallyCloseFile() {
    }
    
    BranchIDLists const*
    OutputModuleBase::branchIDLists() {
      if(!droppedBranchIDToKeptBranchID_.empty()) {
        // Make a private copy of the BranchIDLists.
        *branchIDLists_ = *origBranchIDLists_;
        // Check for branches dropped while an EDAlias was kept.
        for(BranchIDList& branchIDList : *branchIDLists_) {
          for(BranchID::value_type& branchID : branchIDList) {
            // Replace BranchID of each dropped branch with that of the kept alias, so the alias branch will have the product ID of the original branch.
            std::map<BranchID::value_type, BranchID::value_type>::const_iterator iter = droppedBranchIDToKeptBranchID_.find(branchID);
            if(iter != droppedBranchIDToKeptBranchID_.end()) {
              branchID = iter->second;
            }
          }
        }
        return branchIDLists_.get();
      }
      return origBranchIDLists_;
    }
    
    ThinnedAssociationsHelper const*
    OutputModuleBase::thinnedAssociationsHelper() const {
      return thinnedAssociationsHelper_.get();
    }

    ModuleDescription const&
    OutputModuleBase::description() const {
      return moduleDescription_;
    }
    
    bool
    OutputModuleBase::selected(BranchDescription const& desc) const {
      return productSelector_.selected(desc);
    }
    
    void
    OutputModuleBase::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setUnknown();
      descriptions.addDefault(desc);
    }
    
    void
    OutputModuleBase::fillDescription(ParameterSetDescription& desc) {
      ProductSelectorRules::fillDescription(desc, "outputCommands");
      EventSelector::fillDescription(desc);
    }
    
    void
    OutputModuleBase::prevalidate(ConfigurationDescriptions& ) {
    }
    
    
    static const std::string kBaseType("OutputModule");
    const std::string&
    OutputModuleBase::baseType() {
      return kBaseType;
    }
    
    void
    OutputModuleBase::setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                                        bool anyProductProduced) {
      selector_config_id_ = detail::registerProperSelectionInfo(getParameterSet(selector_config_id_),
                                                                description().moduleLabel(),
                                                                outputModulePathPositions,
                                                                anyProductProduced);
    }
  }
}
