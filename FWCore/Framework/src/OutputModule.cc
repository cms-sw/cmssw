/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/OutputModule.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/OutputModuleDescription.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/src/CPCSentry.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include <cassert>

namespace edm {

  // -------------------------------------------------------
  OutputModule::OutputModule(ParameterSet const& pset) :
    maxEvents_(-1),
    remainingEvents_(maxEvents_),
    keptProducts_(),
    hasNewlyDroppedBranch_(),
    process_name_(),
    productSelectorRules_(pset, "outputCommands", "OutputModule"),
    productSelector_(),
    moduleDescription_(),
    current_context_(nullptr),
    wantAllEvents_(false),
    selectors_(),
    selector_config_id_(),
    droppedBranchIDToKeptBranchID_(),
    keptBranchIDToDroppedBranchID_(),
    branchIDLists_(new BranchIDLists),
    origBranchIDLists_(nullptr),
    branchParents_(),
    branchChildren_() {

    hasNewlyDroppedBranch_.fill(false);

    Service<service::TriggerNamesService> tns;
    process_name_ = tns->getProcessName();

    ParameterSet selectevents =
      pset.getUntrackedParameterSet("SelectEvents", ParameterSet());

    selectevents.registerIt(); // Just in case this PSet is not registered

    selector_config_id_ = selectevents.id();
    wantAllEvents_ = detail::configureEventSelector(selectevents,
                                                    process_name_,
                                                    getAllTriggerNames(),
                                                    selectors_);
  }

  void OutputModule::configure(OutputModuleDescription const& desc) {
    remainingEvents_ = maxEvents_ = desc.maxEvents_;
    origBranchIDLists_ = desc.branchIDLists_;
  }

  void OutputModule::selectProducts(ProductRegistry const& preg) {
    if(productSelector_.initialized()) return;
    productSelector_.initialize(productSelectorRules_, preg.allBranchDescriptions());

    // TODO: See if we can collapse keptProducts_ and productSelector_ into a
    // single object. See the notes in the header for ProductSelector
    // for more information.

    std::map<BranchID, BranchDescription const*> trueBranchIDToKeptBranchDesc;

    for(auto const& it : preg.productList()) {
      BranchDescription const& desc = it.second;
      if(desc.transient()) {
        // if the class of the branch is marked transient, output nothing
      } else if(!desc.present() && !desc.produced()) {
        // else if the branch containing the product has been previously dropped,
        // output nothing
      } else if(selected(desc)) {
        // else if the branch has been selected, put it in the list of selected branches.
        if(desc.produced()) {
          // First we check if an equivalent branch has already been selected due to an EDAlias.
          // We only need the check for products produced in this process.
          BranchID const& trueBranchID = desc.originalBranchID();
          std::map<BranchID, BranchDescription const*>::const_iterator iter = trueBranchIDToKeptBranchDesc.find(trueBranchID);
          if(iter != trueBranchIDToKeptBranchDesc.end()) {
             throw edm::Exception(errors::Configuration, "Duplicate Output Selection")
               << "Two (or more) equivalent branches have been selected for output.\n"
               << "#1: " << BranchKey(desc) << "\n" 
               << "#2: " << BranchKey(*iter->second) << "\n" 
               << "Please drop at least one of them.\n";
          }
          trueBranchIDToKeptBranchDesc.insert(std::make_pair(trueBranchID, &desc));
        }
        switch (desc.branchType()) {
          case InEvent:
          {
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
      } else {
        // otherwise, output nothing,
        // and mark the fact that there is a newly dropped branch of this type.
        hasNewlyDroppedBranch_[desc.branchType()] = true;
      }
    }
    // Now fill in a mapping needed in the case that a branch was dropped while its EDAlias was kept.
    for(auto const& it : preg.productList()) {
      BranchDescription const& desc = it.second;
      if(!desc.produced() || desc.isAlias()) continue;
      BranchID const& branchID = desc.branchID();
      std::map<BranchID, BranchDescription const*>::const_iterator iter = trueBranchIDToKeptBranchDesc.find(branchID);
      if(iter != trueBranchIDToKeptBranchDesc.end()) {
        // This branch, produced in this process, or an alias of it, was persisted.
        BranchID const& keptBranchID = iter->second->branchID();
        if(keptBranchID != branchID) {
          // An EDAlias branch was persisted.
          droppedBranchIDToKeptBranchID_.insert(std::make_pair(branchID.id(), keptBranchID.id()));
          keptBranchIDToDroppedBranchID_.insert(std::make_pair(keptBranchID.id(), branchID.id()));
        }
      }
    }
  }

  OutputModule::~OutputModule() { }

  void OutputModule::doBeginJob() {
    this->beginJob();
  }

  void OutputModule::doEndJob() {
    endJob();
  }


  Trig OutputModule::getTriggerResults(EventPrincipal const& ep) const {
    return selectors_.getOneTriggerResults(ep);  }

  namespace {
  }

  bool
  OutputModule::doEvent(EventPrincipal const& ep,
                        EventSetup const&,
                        CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    detail::TRBESSentry products_sentry(selectors_);

    FDEBUG(2) << "writeEvent called\n";

    if(!wantAllEvents_) {
      if(!selectors_.wantEvent(ep)) {
        return true;
      }
    }
    write(ep);
    updateBranchParents(ep);
    if(remainingEvents_ > 0) {
      --remainingEvents_;
    }
    return true;
  }

//   bool OutputModule::wantEvent(Event const& ev)
//   {
//     getTriggerResults(ev);
//     bool eventAccepted = false;

//     typedef std::vector<NamedEventSelector>::const_iterator iter;
//     for(iter i = selectResult_.begin(), e = selectResult_.end();
//          !eventAccepted && i != e; ++i)
//       {
//         eventAccepted = i->acceptEvent(*prods_);
//       }

//     FDEBUG(2) << "Accept event " << ep.id() << " " << eventAccepted << "\n";
//     return eventAccepted;
//   }

  bool
  OutputModule::doBeginRun(RunPrincipal const& rp,
                                EventSetup const&,
                                CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    FDEBUG(2) << "beginRun called\n";
    beginRun(rp);
    return true;
  }

  bool
  OutputModule::doEndRun(RunPrincipal const& rp,
                              EventSetup const&,
                              CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    FDEBUG(2) << "endRun called\n";
    endRun(rp);
    return true;
  }

  void
  OutputModule::doWriteRun(RunPrincipal const& rp) {
    FDEBUG(2) << "writeRun called\n";
    writeRun(rp);
  }

  bool
  OutputModule::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                            EventSetup const&,
                                            CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    FDEBUG(2) << "beginLuminosityBlock called\n";
    beginLuminosityBlock(lbp);
    return true;
  }

  bool
  OutputModule::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                          EventSetup const&,
                                          CurrentProcessingContext const* cpc) {
    detail::CPCSentry sentry(current_context_, cpc);
    FDEBUG(2) << "endLuminosityBlock called\n";
    endLuminosityBlock(lbp);
    return true;
  }

  void OutputModule::doWriteLuminosityBlock(LuminosityBlockPrincipal const& lbp) {
    FDEBUG(2) << "writeLuminosityBlock called\n";
    writeLuminosityBlock(lbp);
  }

  void OutputModule::doOpenFile(FileBlock const& fb) {
    openFile(fb);
  }

  void OutputModule::doRespondToOpenInputFile(FileBlock const& fb) {
    respondToOpenInputFile(fb);
  }

  void OutputModule::doRespondToCloseInputFile(FileBlock const& fb) {
    respondToCloseInputFile(fb);
  }

  void
  OutputModule::doPreForkReleaseResources() {
    preForkReleaseResources();
  }

  void
  OutputModule::doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
    postForkReacquireResources(iChildIndex, iNumberOfChildren);
  }

  void OutputModule::maybeOpenFile() {
    if(!isFileOpen()) reallyOpenFile();
  }

  void OutputModule::doCloseFile() {
    if(isFileOpen()) {
      fillDependencyGraph();
      reallyCloseFile();
      branchParents_.clear();
      branchChildren_.clear();
    }
  }

  void OutputModule::reallyCloseFile() {
  }

  BranchIDLists const*
  OutputModule::branchIDLists() const {
    if(!droppedBranchIDToKeptBranchID_.empty()) {
      // Make a private copy of the BranchIDLists.
      *branchIDLists_ = *origBranchIDLists_;
      // Check for branches dropped while an EDAlias was kept.
      for(BranchIDList& branchIDList : *branchIDLists_) {
        for(BranchID::value_type& branchID : branchIDList) {
          // Replace BranchID of each kept alias branch with zero, so only the product ID of the original branch will be accessible.
          std::map<BranchID::value_type, BranchID::value_type>::const_iterator kiter = keptBranchIDToDroppedBranchID_.find(branchID);
          if(kiter != keptBranchIDToDroppedBranchID_.end()) {
            branchID = 0;
          }
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

  CurrentProcessingContext const*
  OutputModule::currentContext() const {
    return current_context_;
  }

  ModuleDescription const&
  OutputModule::description() const {
    return moduleDescription_;
  }

  bool
  OutputModule::selected(BranchDescription const& desc) const {
    return productSelector_.selected(desc);
  }

  void
  OutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }
  
  void
  OutputModule::fillDescription(ParameterSetDescription& desc) {
    ProductSelectorRules::fillDescription(desc, "outputCommands");
    EventSelector::fillDescription(desc);
  }
  
  void
  OutputModule::prevalidate(ConfigurationDescriptions& ) {
  }
  

  static const std::string kBaseType("OutputModule");
  const std::string&
  OutputModule::baseType() {
    return kBaseType;
  }
  
  void
  OutputModule::setEventSelectionInfo(std::map<std::string, std::vector<std::pair<std::string, int> > > const& outputModulePathPositions,
                                      bool anyProductProduced) {
    selector_config_id_ = detail::registerProperSelectionInfo(getParameterSet(selector_config_id_),
                                      description().moduleLabel(),
                                                      outputModulePathPositions,
                                                      anyProductProduced);
  }

  void
  OutputModule::updateBranchParents(EventPrincipal const& ep) {
    for(EventPrincipal::const_iterator i = ep.begin(), iEnd = ep.end(); i != iEnd; ++i) {
      if((*i) && (*i)->productProvenancePtr() != 0) {
        BranchID const& bid = (*i)->branchDescription().branchID();
        BranchParents::iterator it = branchParents_.find(bid);
        if(it == branchParents_.end()) {
          it = branchParents_.insert(std::make_pair(bid, std::set<ParentageID>())).first;
        }
        it->second.insert((*i)->productProvenancePtr()->parentageID());
        branchChildren_.insertEmpty(bid);
      }
    }
  }

  void
  OutputModule::fillDependencyGraph() {
    for(BranchParents::const_iterator i = branchParents_.begin(), iEnd = branchParents_.end();
        i != iEnd; ++i) {
      BranchID const& child = i->first;
      std::set<ParentageID> const& eIds = i->second;
      for(std::set<ParentageID>::const_iterator it = eIds.begin(), itEnd = eIds.end();
          it != itEnd; ++it) {
        Parentage entryDesc;
        ParentageRegistry::instance()->getMapped(*it, entryDesc);
        std::vector<BranchID> const& parents = entryDesc.parents();
        for(std::vector<BranchID>::const_iterator j = parents.begin(), jEnd = parents.end();
          j != jEnd; ++j) {
          branchChildren_.insertChild(*j, child);
        }
      }
    }
  }
}
