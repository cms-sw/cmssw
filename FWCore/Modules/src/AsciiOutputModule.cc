/*----------------------------------------------------------------------
$Id: AsciiOutputModule.cc,v 1.9 2007/03/04 05:53:06 wmtan Exp $
----------------------------------------------------------------------*/

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ostream>

#include "FWCore/Modules/src/AsciiOutputModule.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  AsciiOutputModule::AsciiOutputModule(ParameterSet const& pset, std::ostream* os) :
    OutputModule(pset),
    prescale_(pset.getUntrackedParameter("prescale", 1U)),
    verbosity_(pset.getUntrackedParameter("verbosity", 1U)),
    counter_(0),
    pout_(os) {
    if (prescale_ == 0) prescale_ = 1;
  }

  AsciiOutputModule::~AsciiOutputModule() {
    *pout_ << ">>> processed " << counter_ << " events" << std::endl;
  }

  void
  AsciiOutputModule::write(const EventPrincipal& e) {


    if ((++counter_ % prescale_) != 0 || verbosity_ <= 0) return;

    //  const Run & run = evt.getRun(); // this is still unused
    *pout_ << ">>> processing event # " << e.id() <<" time " <<e.time().value()
           << std::endl;

    if (verbosity_ <= 1) return;

    // Write out non-EDProduct contents...

    // ... list of process-names
    for (ProcessHistory::const_iterator it = e.processHistory().begin(), itEnd = e.processHistory().end();
        it != itEnd; ++it) {
      *pout_ << it->processName() << " ";
    }

    // ... collision id
    *pout_ << '\n' << e.id() << '\n';
    
    // Loop over products, and write some output for each...

    std::vector<Provenance const *> provs;
    e.getAllProvenance(provs);
    for(std::vector<Provenance const *>::const_iterator i = provs.begin(),
	 iEnd = provs.end();
	 i != iEnd;
	 ++i) {
      BranchDescription const& desc = (*i)->product();
      if (selected(desc)) {
        *pout_ << **i << '\n';
      }
    }
  }
}
