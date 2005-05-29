/*----------------------------------------------------------------------
$Id: AsciiOutputModule.cc,v 1.1 2005/05/28 05:10:04 wmtan Exp $
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

  AsciiOutputModule::AsciiOutputModule(ParameterSet const&, std::ostream* os) :
    OutputModule(),
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

    std::copy(e.begin(),
	      e.end(),
	      std::ostream_iterator<EventPrincipal::GroupVec::value_type>(*pout_, "\n"));
  }
}
