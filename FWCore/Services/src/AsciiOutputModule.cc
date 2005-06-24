/*----------------------------------------------------------------------
$Id: AsciiOutputModule.cc,v 1.3 2005/06/23 04:33:54 wmtan Exp $
----------------------------------------------------------------------*/

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ostream>

#include "FWCore/FWCoreServices/src/AsciiOutputModule.h"
#include "FWCore/CoreFramework/interface/EventPrincipal.h"
#include "FWCore/CoreFramework/interface/OutputModule.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  AsciiOutputModule::AsciiOutputModule(ParameterSet const& pset, std::ostream* os) :
    OutputModule(pset.getUntrackedParameter("select", ParameterSet())),
    prescale_(pset.getUntrackedParameter("prescale", 1)),
    verbosity_(pset.getUntrackedParameter("verbosity", 1)),
    counter_(0),
    pout_(os)
  {}

  AsciiOutputModule::~AsciiOutputModule() {
    *pout_ << ">>> processed " << counter_ << " events" << std::endl;
  }

  void
  AsciiOutputModule::write(const EventPrincipal& e) {


    if ((++counter_ % prescale_) != 0 || verbosity_ <= 0) return;

    //  const Run & run = evt.getRun(); // this is still unused
    *pout_ << ">>> processing event # " << e.id() << std::endl;

    if (verbosity_ <= 1) return;

    // Write out non-EDProduct contents...

    // ... list of process-names
    std::copy(e.beginProcess(),
	      e.endProcess(),
	      std::ostream_iterator<EventPrincipal::ProcessNameList::value_type>(*pout_, " "));

    // ... collision id
    *pout_ << '\n' << e.id() << '\n';
    
    // Loop over groups, and write some output for each...

    for(EventPrincipal::const_iterator i = e.begin(); i != e.end(); ++i) {
      Provenance const& prov = *(*i)->provenance();
      if (selected(prov)) {
        *pout_ << *i << '\n';
      }
    }
  }
}
