/*----------------------------------------------------------------------
$Id: OutputModule.cc,v 1.21 2006/09/09 06:38:56 afaq Exp $
----------------------------------------------------------------------*/

#include <vector>
#include <iostream>

#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/BranchDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/CPCSentry.h"

namespace edm
{
  // This grotesque little function exists just to allow calling of
  // ConstProductRegistry::allBranchDescriptions in the context of
  // OutputModule's initialization list, rather than in the body of
  // the constructor.

  std::vector<edm::BranchDescription const*>
  getAllBranchDescriptions()
  {
    edm::Service<edm::ConstProductRegistry> reg;
    return reg->allBranchDescriptions();
  }

  std::vector<std::string> const& getAllTriggerNames()
  {
    edm::Service<edm::service::TriggerNamesService> tns;
    return tns->getTrigPaths();
  }

//}


//namespace edm {
  OutputModule::OutputModule(ParameterSet const& pset) : 
    nextID_(),
    descVec_(),
    droppedVec_(),
    process_name_(Service<service::TriggerNamesService>()->getProcessName()),
    groupSelector_(pset),
    eventSelector_(pset,process_name_,
		   getAllTriggerNames()),
    // use this temporarily - can only apply event selection to current
    // process name
    selectResult_(eventSelector_.getProcessName()),
    current_context_(0)
  {
  }

  void OutputModule::selectProducts() {
    if (groupSelector_.initialized()) return;
    groupSelector_.initialize(getAllBranchDescriptions());
    Service<ConstProductRegistry> reg;
    nextID_ = reg->nextID();

    // TODO: See if we can collapse descVec_ and groupSelector_ into a
    // single object. See the notes in the header for GroupSelector
    // for more information.

    ProductRegistry::ProductList::const_iterator it  = 
      reg->productList().begin();
    ProductRegistry::ProductList::const_iterator end = 
      reg->productList().end();

    for ( ; it != end; ++it) {
      BranchDescription const& desc = it->second;
      if(!desc.provenancePresent() & !desc.produced()) {
        // If the branch containing the provenance has been previously dropped,
        // and the product has not been produced again, output nothing
        continue;
      } else if(desc.transient()) {
        // else if the class of the branch is marked transient, drop the product branch
        droppedVec_.push_back(&desc);
        continue;
      } else if(!desc.present() & !desc.produced()) {
        // else if the branch containing the product has been previously dropped,
        // and the product has not been produced again, drop the product branch again.
        droppedVec_.push_back(&desc);
      } else if (selected(desc)) {
        // else if the branch has been selected, put it in the list of selected branches
        descVec_.push_back(&desc);
      } else {
        // otherwise, drop the product branch.
        droppedVec_.push_back(&desc);
      }
    }
  }

  OutputModule::~OutputModule() { }

  void OutputModule::beginJob(EventSetup const&) { }

  void OutputModule::endJob() { }

 const Trig& OutputModule::getTrigMask(EventPrincipal const& ep) const
  {
    if (! prod_.isValid())
    {
      // use module description and const_cast unless interface to
      // event is changed to just take a const EventPrincipal
      Event e(const_cast<EventPrincipal&>(ep), *current_md_);
      try {
         e.get(selectResult_, prod_);
      } catch(cms::Exception& e){
         FDEBUG(2) << e.what();
         //prod_ stays empty
      }
    }
    return  prod_;
  }

  void OutputModule::writeEvent(EventPrincipal const& ep,
                                ModuleDescription const& md,
                                CurrentProcessingContext const* c)
  {
    detail::CPCSentry sentry(current_context_, c);
    //Save the current Mod Desc
    current_md_ = &md;

    FDEBUG(2) << "writeEvent called\n";
    if(eventSelector_.wantAll() || wantEvent(ep)) {
         write(ep);
    }
    //Clean up the TriggerResult handle immediately
    // for next event should get it empty (inValid() returning true)
    prod_ = edm::Handle<edm::TriggerResults>();
  }

  bool OutputModule::wantEvent(EventPrincipal const& ep)
  {
    // this implementation cannot deal with doing event selection
    // based on any previous TriggerResults.  It can only select
    // based on a TriggerResult made in the current process.

    // use module description and const_cast unless interface to
    // event is changed to just take a const EventPrincipal

    getTrigMask(ep);

    bool rc = eventSelector_.acceptEvent(*prod_);
    FDEBUG(2) << "Accept event " << ep.id() << " " << rc << "\n";
    FDEBUG(2) << "Mask: " << *prod_ << "\n";
    return rc;
  }

  void OutputModule::doBeginRun(RunPrincipal const& rp,
                                ModuleDescription const& md,
                                CurrentProcessingContext const* c)
  {
    detail::CPCSentry sentry(current_context_, c);
    //Save the current Mod Desc
    current_md_ = &md;

    FDEBUG(2) << "beginRun called\n";
    beginRun(rp);
  }

  void OutputModule::doEndRun(RunPrincipal const& rp,
                                ModuleDescription const& md,
                                CurrentProcessingContext const* c)
  {
    detail::CPCSentry sentry(current_context_, c);
    //Save the current Mod Desc
    current_md_ = &md;

    FDEBUG(2) << "endRun called\n";
    endRun(rp);
  }

  void OutputModule::doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                ModuleDescription const& md,
                                CurrentProcessingContext const* c)
  {
    detail::CPCSentry sentry(current_context_, c);
    //Save the current Mod Desc
    current_md_ = &md;

    FDEBUG(2) << "beginLuminosityBlock called\n";
    beginLuminosityBlock(lbp);
  }

  void OutputModule::doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                ModuleDescription const& md,
                                CurrentProcessingContext const* c)
  {
    detail::CPCSentry sentry(current_context_, c);
    //Save the current Mod Desc
    current_md_ = &md;

    FDEBUG(2) << "endLuminosityBlock called\n";
    endLuminosityBlock(lbp);
  }

  CurrentProcessingContext const*
  OutputModule::currentContext() const
  {
    return current_context_;
  }

  bool OutputModule::selected(BranchDescription const& desc) const
  {
    return groupSelector_.selected(desc);
  }

  unsigned long OutputModule::nextID() const 
  {
    return nextID_;
  }
}
