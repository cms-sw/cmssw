#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"

namespace edm {
  EventPrincipal::EventPrincipal(EventID const& id,
	Timestamp const& time,
	ProductRegistry const& reg,
        boost::shared_ptr<LuminosityBlockPrincipal> lbp,
	ProcessConfiguration const& pc,
	ProcessHistoryID const& hist,
	boost::shared_ptr<DelayedReader> rtrv) :
	  Base(reg, pc, hist, rtrv),
	  aux_(id, time, lbp->luminosityBlock()),
	  luminosityBlockPrincipal_(lbp),
	  unscheduledHandler_(),
	  provenanceFiller_() {}

  EventPrincipal::EventPrincipal(EventID const& id,
	Timestamp const& time,
	ProductRegistry const& reg,
	LuminosityBlockNumber_t lumi,
	ProcessConfiguration const& pc,
	ProcessHistoryID const& hist,
	boost::shared_ptr<DelayedReader> rtrv) :
	  Base(reg, pc, hist, rtrv),
	  aux_(id, time, lumi),
	  luminosityBlockPrincipal_(new LuminosityBlockPrincipal(lumi, reg, id.run(), pc)),
	  unscheduledHandler_(),
	  provenanceFiller_() {}

  RunPrincipal const&
  EventPrincipal::runPrincipal() const {
    return luminosityBlockPrincipal().runPrincipal();
  }

  RunPrincipal &
  EventPrincipal::runPrincipal() {
    return luminosityBlockPrincipal().runPrincipal();
  }

  void
  EventPrincipal::setUnscheduledHandler(boost::shared_ptr<UnscheduledHandler> iHandler) {
    unscheduledHandler_ = iHandler;
    provenanceFiller_ = boost::shared_ptr<EPEventProvenanceFiller>(
	new EPEventProvenanceFiller(unscheduledHandler_, const_cast<EventPrincipal *>(this)));
    setUnscheduled();
  }

  bool
  EventPrincipal::unscheduledFill(Group const& group) const {
    if (unscheduledHandler_ &&
	unscheduledHandler_->tryToFill(group.provenance(), *const_cast<EventPrincipal *>(this))) {
      //see if product actually retrieved.
      if(!group.product()) {
        throw edm::Exception(errors::ProductNotFound, "InaccessibleProduct")
          <<"product not accessible\n" << group.provenance();
      }
      return true;
    }
    return false;
  }

  bool
  EventPrincipal::fillAndMatchSelector(Provenance& prov, SelectorBase const& selector) const {
    ProvenanceAccess provAccess(&prov, provenanceFiller_.get());
    return (selector.match(provAccess));
  }

}
