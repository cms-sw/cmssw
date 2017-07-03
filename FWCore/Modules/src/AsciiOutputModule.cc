/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <algorithm>
#include <iterator>
#include <ostream>
#include <iostream>
#include <string>
#include "FWCore/Framework/interface/global/OutputModule.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {

  class ModuleCallingContext;

  class AsciiOutputModule : public global::OutputModule<> {
  public:
    // We do not take ownership of passed stream.
    explicit AsciiOutputModule(ParameterSet const& pset);
    ~AsciiOutputModule() override;
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void write(EventForOutput const& e) override;
    void writeLuminosityBlock(LuminosityBlockForOutput const&) override {}
    void writeRun(RunForOutput const&) override {}
    int prescale_;
    int verbosity_;
    int counter_;
  };

  AsciiOutputModule::AsciiOutputModule(ParameterSet const& pset) :
    global::OutputModuleBase(pset),
    global::OutputModule<>(pset),
    prescale_(pset.getUntrackedParameter<unsigned int>("prescale")),
    verbosity_(pset.getUntrackedParameter<unsigned int>("verbosity")),
    counter_(0) {
     if (prescale_ == 0) prescale_ = 1;
  }

  AsciiOutputModule::~AsciiOutputModule() {
    LogAbsolute("AsciiOut") << ">>> processed " << counter_ << " events" << std::endl;
  }

  void
  AsciiOutputModule::write(EventForOutput const& e) {

    if ((++counter_ % prescale_) != 0 || verbosity_ <= 0) return;

    // RunForOutput const& run = evt.getRun(); // this is still unused
    LogAbsolute("AsciiOut")<< ">>> processing event # " << e.id() << " time " << e.time().value() << std::endl;

    if (verbosity_ <= 1) return;

    // Write out non-EDProduct contents...

    // ... list of process-names
    for (auto const& process : e.processHistory()) {
      LogAbsolute("AsciiOut") << process.processName() << " ";
    }

    // ... collision id
    LogAbsolute("AsciiOut") << '\n' << e.id() << '\n';

    // Loop over products, and write some output for each...

    std::vector<Provenance const*> provs;
    e.getAllProvenance(provs);
    for(auto const& prov : provs) {
      BranchDescription const& desc = prov->branchDescription();
      if (selected(desc)) {
        LogAbsolute("AsciiOut") << *prov << '\n';
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
