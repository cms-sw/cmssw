
#include "FWCore/Framework/interface/one/OutputModule.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

#include <cassert>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>

using namespace edm;

using Trig = detail::TriggerResultsBasedEventSelector::handle_t;

namespace {

  void printBits(unsigned char c) {
    //cout << "HEX: "<< "0123456789ABCDEF"[((c >> 4) & 0xF)] << std::endl;

    for (int i = 7; i >= 0; --i) {
      int bit = ((c >> i) & 1);
      std::cout << " " << bit;
    }
  }

  void packIntoString(std::vector<unsigned char> const& source, std::vector<unsigned char>& package) {
    unsigned int packInOneByte = 4;
    unsigned int sizeOfPackage = 1 + ((source.size() - 1) / packInOneByte);  //Two bits per HLT
    if (source.size() == 0)
      sizeOfPackage = 0;

    package.resize(sizeOfPackage);
    memset(&package[0], 0x00, sizeOfPackage);

    for (unsigned int i = 0; i != source.size(); ++i) {
      unsigned int whichByte = i / packInOneByte;
      unsigned int indxWithinByte = i % packInOneByte;
      package[whichByte] = package[whichByte] | (source[i] << (indxWithinByte * 2));
    }
    //for (unsigned int i=0; i !=package.size() ; ++i)
    //   printBits(package[i]);
    std::cout << std::endl;
  }

}  // namespace
namespace edmtest {

  class TestOutputModule : public edm::one::OutputModule<> {
  public:
    explicit TestOutputModule(edm::ParameterSet const&);
    virtual ~TestOutputModule();

  private:
    Trig getTriggerResults(EDGetTokenT<TriggerResults> const& token, EventForOutput const& e) const;

    virtual void write(edm::EventForOutput const& e) override;
    virtual void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) override;
    virtual void writeRun(edm::RunForOutput const&) override;
    virtual void endJob() override;

    std::string name_;
    int bitMask_;
    std::vector<unsigned char> hltbits_;
    bool expectTriggerResults_;
    edm::EDGetTokenT<edm::TriggerResults> resultsToken_;
  };

  // -----------------------------------------------------------------

  TestOutputModule::TestOutputModule(edm::ParameterSet const& ps)
      : edm::one::OutputModuleBase(ps),
        edm::one::OutputModule<>(ps),
        name_(ps.getParameter<std::string>("name")),
        bitMask_(ps.getParameter<int>("bitMask")),
        hltbits_(0),
        expectTriggerResults_(ps.getUntrackedParameter<bool>("expectTriggerResults", true)),
        resultsToken_(consumes(edm::InputTag("TriggerResults"))) {}

  TestOutputModule::~TestOutputModule() {}

  Trig TestOutputModule::getTriggerResults(EDGetTokenT<TriggerResults> const& token,
                                           EventForOutput const& event) const {
    return event.getHandle(token);
  }

  void TestOutputModule::write(edm::EventForOutput const& e) {
    assert(e.moduleCallingContext()->moduleDescription()->moduleLabel() == description().moduleLabel());

    Trig prod;

    // There should not be a TriggerResults object in the event
    // if all three of the following requirements are met:
    //
    //     1.  MakeTriggerResults has not been explicitly set true
    //     2.  There are no filter modules in any path
    //     3.  The input file of the job does not have a TriggerResults object
    //
    // The user of this test module is expected to know
    // whether these conditions are met and let the module know
    // if no TriggerResults object is expected using the configuration
    // file.  In this case, the next few lines of code will abort
    // if a TriggerResults object is found.

    if (!expectTriggerResults_) {
      try {
        prod = getTriggerResults(resultsToken_, e);
        //throw doesn't happen until we dereference
        *prod;
      } catch (const cms::Exception&) {
        // We did not find one as expected, nothing else to test.
        return;
      }
      std::cerr << "\nTestOutputModule::write\n"
                << "Expected there to be no TriggerResults object but we found one" << std::endl;
      abort();
    }

    // Now deal with the other case where we expect the object
    // to be present.

    prod = getTriggerResults(resultsToken_, e);

    std::vector<unsigned char> vHltState;

    std::vector<std::string> hlts = getAllTriggerNames();
    unsigned int hltSize = hlts.size();

    for (unsigned int i = 0; i != hltSize; ++i) {
      vHltState.push_back(((prod->at(i)).state()));
    }

    //Pack into member hltbits_
    packIntoString(vHltState, hltbits_);

    //This is Just a printing code.
    std::cout << "Size of hltbits:" << hltbits_.size() << std::endl;

    char* intp = (char*)&bitMask_;
    bool matched = false;

    for (int i = hltbits_.size() - 1; i != -1; --i) {
      std::cout << std::endl << "Current Bits Mask byte:";
      printBits(hltbits_[i]);
      unsigned char tmp = static_cast<unsigned char>(*(intp + i));
      std::cout << std::endl << "Original Byte:";
      printBits(tmp);
      std::cout << std::endl;

      if (tmp == hltbits_[i])
        matched = true;
    }
    std::cout << "\n";

    if (!matched && hltSize > 0) {
      std::cerr << "\ncfg bitMask is different from event..aborting." << std::endl;

      abort();
    } else
      std::cout << "\nSUCCESS: Found Matching Bits" << std::endl;
  }

  void TestOutputModule::writeLuminosityBlock(edm::LuminosityBlockForOutput const& lb) {
    assert(lb.moduleCallingContext()->moduleDescription()->moduleLabel() == description().moduleLabel());
  }

  void TestOutputModule::writeRun(edm::RunForOutput const& r) {
    assert(r.moduleCallingContext()->moduleDescription()->moduleLabel() == description().moduleLabel());
  }

  void TestOutputModule::endJob() {}
}  // namespace edmtest
using edmtest::TestOutputModule;

DEFINE_FWK_MODULE(TestOutputModule);
