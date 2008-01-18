/*----------------------------------------------------------------------
$Id: AsciiOutputModule.cc,v 1.12 2008/01/14 16:49:22 chrjones Exp $
----------------------------------------------------------------------*/

#include <algorithm>
#include <iterator>
#include <ostream>
#include "FWCore/Modules/src/AsciiOutputModule.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



namespace edm {

  AsciiOutputModule::AsciiOutputModule(ParameterSet const& pset) :
    OutputModule(pset),
    prescale_(pset.getUntrackedParameter("prescale", 1U)),
    verbosity_(pset.getUntrackedParameter("verbosity", 1U)),
    counter_(0)
    {
       if (prescale_ == 0) prescale_ = 1;
     }
    

  AsciiOutputModule::~AsciiOutputModule() {
    edm::LogAbsolute("AsciiOut")<< ">>> processed " << counter_ << " events" << std::endl;
  }

  void
  AsciiOutputModule::write(const EventPrincipal& e) {


    if ((++counter_ % prescale_) != 0 || verbosity_ <= 0) return;

    //  const Run & run = evt.getRun(); // this is still unused
    edm::LogAbsolute("AsciiOut")<< ">>> processing event # " << e.id() <<" time " <<e.time().value()
				<< std::endl;

    if (verbosity_ <= 1) return;

    // Write out non-EDProduct contents...

    // ... list of process-names
    for (ProcessHistory::const_iterator it = e.processHistory().begin(), itEnd = e.processHistory().end();
        it != itEnd; ++it) {
      edm::LogAbsolute("AsciiOut")<< it->processName() << " ";
    }

    // ... collision id
    edm::LogAbsolute("AsciiOut")<< '\n' << e.id() << '\n';
   
    
    // Loop over products, and write some output for each...

    std::vector<Provenance const *> provs;
    e.getAllProvenance(provs);
    for(std::vector<Provenance const *>::const_iterator i = provs.begin(),
	 iEnd = provs.end();
	 i != iEnd;
	 ++i) {
      BranchDescription const& desc = (*i)->product();
      if (selected(desc)) {
	edm::LogAbsolute("AsciiOut")<< **i << '\n';
      }
    }
  }
}
