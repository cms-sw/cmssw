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
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"

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

  AsciiOutputModule::AsciiOutputModule(ParameterSet const& pset)
      : global::OutputModuleBase(pset),
        global::OutputModule<>(pset),
        prescale_(pset.getUntrackedParameter<unsigned int>("prescale")),
        verbosity_(pset.getUntrackedParameter<unsigned int>("verbosity")),
        counter_(0) {
    if (prescale_ == 0)
      prescale_ = 1;
  }

  AsciiOutputModule::~AsciiOutputModule() {
    LogAbsolute("AsciiOut") << ">>> processed " << counter_ << " events" << std::endl;
  }

  void AsciiOutputModule::write(EventForOutput const& e) {
    if ((++counter_ % prescale_) != 0 || verbosity_ <= 0)
      return;

    // RunForOutput const& run = evt.getRun(); // this is still unused
    LogAbsolute("AsciiOut") << ">>> processing event # " << e.id() << " time " << e.time().value() << std::endl;

    if (verbosity_ <= 1)
      return;

    // Write out non-EDProduct contents...

    // ... list of process-names
    for (auto const& process : e.processHistory()) {
      LogAbsolute("AsciiOut") << process.processName() << " ";
    }

    // ... collision id
    LogAbsolute("AsciiOut") << '\n' << e.id() << '\n';

    // Loop over products, and write some output for each...
    Service<ConstProductRegistry> reg;
    for (auto const& prod : reg->productList()) {
      BranchDescription const& desc = prod.second;
      if (selected(desc)) {
        if (desc.isAlias()) {
          LogAbsolute("AsciiOut") << "ModuleLabel " << desc.moduleLabel() << " is an alias for";
        }

        auto const& prov = e.getProvenance(desc.originalBranchID());
        LogAbsolute("AsciiOut") << prov;

        if (verbosity_ > 2) {
          BranchDescription const& desc2 = prov.branchDescription();
          std::string const& process = desc2.processName();
          std::string const& label = desc2.moduleLabel();
          ProcessHistory const& processHistory = e.processHistory();

          for (ProcessConfiguration const& pc : processHistory) {
            if (pc.processName() == process) {
              ParameterSetID const& psetID = pc.parameterSetID();
              pset::Registry const* psetRegistry = pset::Registry::instance();
              ParameterSet const* processPset = psetRegistry->getMapped(psetID);
              if (processPset) {
                if (desc.isAlias()) {
                  LogAbsolute("AsciiOut") << "Alias PSet\n" << processPset->getParameterSet(desc.moduleLabel());
                }
                LogAbsolute("AsciiOut") << processPset->getParameterSet(label) << "\n";
              }
            }
          }
        }
      }
    }
  }

  void AsciiOutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Outputs event information into text file.");
    desc.addUntracked("prescale", 1U)->setComment("prescale factor");
    desc.addUntracked("verbosity", 1U)
        ->setComment(
            "0: no output\n"
            "1: event ID and timestamp only\n"
            "2: provenance for each kept product\n"
            ">2: PSet and provenance for each kept product");
    OutputModule::fillDescription(desc);
    descriptions.add("asciiOutput", desc);
  }
}  // namespace edm

using edm::AsciiOutputModule;
DEFINE_FWK_MODULE(AsciiOutputModule);
