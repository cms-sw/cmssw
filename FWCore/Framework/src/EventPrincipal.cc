#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/UnscheduledHandler.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/src/Group.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "DataFormats/Provenance/interface/Provenance.h"

#include <algorithm>

namespace edm {
  EventPrincipal::EventPrincipal(EventAuxiliary const& aux,
	boost::shared_ptr<ProductRegistry const> reg,
        boost::shared_ptr<LuminosityBlockPrincipal> lbp,
	ProcessConfiguration const& pc,
	ProcessHistoryID const& hist,
	boost::shared_ptr<DelayedReader> rtrv) :
	  Base(reg, pc, hist, rtrv),
	  aux_(aux),
	  luminosityBlockPrincipal_(lbp),
	  unscheduledHandler_() {
	  }

  RunPrincipal const&
  EventPrincipal::runPrincipal() const {
    return luminosityBlockPrincipal().runPrincipal();
  }

  RunPrincipal &
  EventPrincipal::runPrincipal() {
    return luminosityBlockPrincipal().runPrincipal();
  }

  void
  EventPrincipal::addGroup(std::auto_ptr<Provenance> prov) {
    std::auto_ptr<Group> g(new Group(prov));
    addOrReplaceGroup(g);
  }

  void
  EventPrincipal::addOrReplaceGroup(std::auto_ptr<Group> g) {
    Group const* group = getExistingGroup(*g);
    if (group != 0) {
      if(!group->onDemand()) {
        BranchDescription const& bd = group->productDescription();
	throw edm::Exception(edm::errors::InsertFailure,"AlreadyPresent")
	  << "addGroup_: Problem found while adding product provenance, "
	  << "product already exists for ("
	  << bd.friendlyClassName() << ","
	  << bd.moduleLabel() << ","
	  << bd.productInstanceName() << ","
	  << bd.processName()
	  << ")\n";
      }
      replaceGroup(g);
    } else {
      addGroup_(g);
    }
  }

  void
  EventPrincipal::setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler> iHandler) {
    unscheduledHandler_ = iHandler;
  }

  EventSelectionIDVector const&
  EventPrincipal::eventSelectionIDs() const
  {
    return eventHistory_.eventSelectionIDs();
  }

  History const&
  EventPrincipal::history() const
  {
    return eventHistory_;
  }

  bool
  EventPrincipal::unscheduledFill(Provenance const& prov) const {

    // If it is a module already currently running in unscheduled
    // mode, then there is a circular dependency related to which
    // EDProducts modules require and produce.  There is no safe way
    // to recover from this.  Here we check for this problem and throw
    // an exception.
    std::vector<std::string>::const_iterator i =
      find_in_all(moduleLabelsRunning_, prov.moduleLabel());

    if (i != moduleLabelsRunning_.end()) {
      throw edm::Exception(errors::LogicError)
        << "Hit circular dependency while trying to run an unscheduled module.\n"
        << "Current implementation of unscheduled execution cannot always determine\n"
        << "the proper order for module execution.  It is also possible the modules\n"
        << "have a built in circular dependence that will not work with any order.\n"
        << "In the first case, scheduling some or all required modules in paths will help.\n"
        << "In the second case, the modules themselves will have to be fixed.\n";
    }

    moduleLabelsRunning_.push_back(prov.moduleLabel());

    if (unscheduledHandler_) {
      unscheduledHandler_->tryToFill(prov, *const_cast<EventPrincipal *>(this));
    }
    moduleLabelsRunning_.pop_back();
    return true;
  }

  void
  EventPrincipal::setHistory(History const& h) {
    eventHistory_ = h;
  }
}
