/*----------------------------------------------------------------------
$Id: OutputModule.cc,v 1.18 2006/08/24 17:11:52 wmtan Exp $
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

namespace
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

}


namespace edm {
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
      if(desc.transient()) {
        // If the class of the branch is marked transient, output nothing
        continue;
      } else if(!desc.provenancePresent() & !desc.produced()) {
        // else if the branch containing the provenance has been previously dropped,
        // and the product has not been produced again, output nothing
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

  void OutputModule::writeEvent(EventPrincipal const& ep,
				ModuleDescription const& md,
				CurrentProcessingContext const* c)
  {
    detail::CPCSentry sentry(current_context_, c);
    FDEBUG(2) << "writeEvent called\n";
    if(eventSelector_.wantAll() || wantEvent(ep,md)) write(ep);
  }

  bool OutputModule::wantEvent(EventPrincipal const& ep,
			       ModuleDescription const& md)
  {
    // this implementation cannot deal with doing event selection
    // based on any previous TriggerResults.  It can only select
    // based on a TriggerResult made in the current process.

    // use module description and const_cast unless interface to
    // event is changed to just take a const EventPrincipal
    Event e(const_cast<EventPrincipal&>(ep),md);
    typedef edm::Handle<edm::TriggerResults> Trig;
    Trig prod;
    e.get(selectResult_,prod);
    bool rc = eventSelector_.acceptEvent(*prod);
	FDEBUG(2) << "Accept event " << ep.id() << " " << rc << "\n";
	FDEBUG(2) << "Mask: " << *prod << "\n";
	return rc;
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
