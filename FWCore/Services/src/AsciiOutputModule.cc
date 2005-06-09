/*----------------------------------------------------------------------
$Id: AsciiOutputModule.cc,v 1.1 2005/05/29 02:29:54 wmtan Exp $
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
    OutputModule(pset),
    pout_(os)
  {}

  AsciiOutputModule::~AsciiOutputModule() {}

  void
  AsciiOutputModule::write(const EventPrincipal& e) {
    // Write out non-EDProduct contents...

    // ... list of process-names
    std::copy(e.beginProcess(),
	      e.endProcess(),
	      std::ostream_iterator<EventPrincipal::ProcessNameList::value_type>(*pout_, " "));

    // ... collision id
    *pout_ << '\n' << e.ID() << '\n';
    
    // Loop over groups, and write some output for each...
//     EventPrincipal::const_iterator it(e.begin());
//     EventPrincipal::const_iterator end(e.end());

    for(EventPrincipal::const_iterator i = e.begin(); i != e.end(); ++i) {
      Provenance const& prov = *(*i)->provenance();
      std::string const& ml = prov.module.module_label;
      if (selected(ml)) {
        *pout_ << *i << '\n';
      }
    }
  }
}
