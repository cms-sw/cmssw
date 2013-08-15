/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <algorithm>
#include <iterator>
#include <ostream>
#include <iostream>
#include <string>
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {
  class AsciiOutputModule : public OutputModule {
  public:
    // We do not take ownership of passed stream.
    explicit AsciiOutputModule(ParameterSet const& pset);
    virtual ~AsciiOutputModule();
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void write(EventPrincipal const& e, ModuleCallingContext const*) override;
    virtual void writeLuminosityBlock(LuminosityBlockPrincipal const&, ModuleCallingContext const*) override {}
    virtual void writeRun(RunPrincipal const&, ModuleCallingContext const*) override {}
    int prescale_;
    int verbosity_;
    int counter_;
  };

  AsciiOutputModule::AsciiOutputModule(ParameterSet const& pset) :
    OutputModule(pset),
    prescale_(pset.getUntrackedParameter<unsigned int>("prescale")),
    verbosity_(pset.getUntrackedParameter<unsigned int>("verbosity")),
    counter_(0) {
     if (prescale_ == 0) prescale_ = 1;
  }

  AsciiOutputModule::~AsciiOutputModule() {
    LogAbsolute("AsciiOut") << ">>> processed " << counter_ << " events" << std::endl;
  }

  void
  AsciiOutputModule::write(EventPrincipal const& e, ModuleCallingContext const*) {

    if ((++counter_ % prescale_) != 0 || verbosity_ <= 0) return;

    // Run const& run = evt.getRun(); // this is still unused
    LogAbsolute("AsciiOut")<< ">>> processing event # " << e.id() << " time " << e.time().value() << std::endl;

    if (verbosity_ <= 1) return;

    // Write out non-EDProduct contents...

    // ... list of process-names
    for (ProcessHistory::const_iterator it = e.processHistory().begin(), itEnd = e.processHistory().end();
        it != itEnd; ++it) {
      LogAbsolute("AsciiOut") << it->processName() << " ";
    }

    // ... collision id
    LogAbsolute("AsciiOut") << '\n' << e.id() << '\n';

    // Loop over products, and write some output for each...

    std::vector<Provenance const*> provs;
    e.getAllProvenance(provs);
    for(std::vector<Provenance const*>::const_iterator i = provs.begin(),
         iEnd = provs.end();
         i != iEnd;
         ++i) {
      BranchDescription const& desc = (*i)->product();
      if (selected(desc)) {
        LogAbsolute("AsciiOut") << **i << '\n';
      }
    }
  }

  void
  AsciiOutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Outputs event information into text file.");
    desc.addUntracked("prescale", 1U)
        ->setComment("prescale factor");
    desc.addUntracked("verbosity", 1U)
        ->setComment("0: no output\n"
                     "1: event ID and timestamp only\n"
                     ">1: full output");
    OutputModule::fillDescription(desc);
    descriptions.add("asciiOutput", desc);
  }
}

using edm::AsciiOutputModule;
DEFINE_FWK_MODULE(AsciiOutputModule);
