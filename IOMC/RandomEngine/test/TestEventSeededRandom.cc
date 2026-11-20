// Prints the first random numbers of each event and of each luminosity block,
// keyed by the EventID, so that jobs with different stream counts can be compared.

#include "FWCore/AbstractServices/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CLHEP/Random/RandomEngine.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

namespace {
  std::string draws(CLHEP::HepRandomEngine& engine, unsigned int count) {
    std::ostringstream text;
    text << std::setprecision(17);
    for (unsigned int i = 0; i < count; ++i) {
      text << " " << engine.flat();
    }
    return text.str();
  }
}  // namespace

class TestEventSeededRandom : public edm::global::EDAnalyzer<edm::LuminosityBlockCache<int>> {
public:
  explicit TestEventSeededRandom(edm::ParameterSet const& pset)
      : draws_(pset.getUntrackedParameter<unsigned int>("draws")) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<unsigned int>("draws", 4);
    descriptions.addDefault(desc);
  }

  void analyze(edm::StreamID streamID, edm::Event const& event, edm::EventSetup const&) const override {
    edm::Service<edm::RandomNumberGenerator> rng;
    std::ostringstream line;
    line << "EVENTRANDOM " << event.id().run() << " " << event.luminosityBlock() << " " << event.id().event() << " "
         << moduleDescription().moduleLabel() << draws(rng->getEngine(streamID), draws_) << "\n";
    // One write per line, so that the lines of concurrent events do not mix.
    std::cout << line.str() << std::flush;
  }

  std::shared_ptr<int> globalBeginLuminosityBlock(edm::LuminosityBlock const& lumi,
                                                  edm::EventSetup const&) const override {
    edm::Service<edm::RandomNumberGenerator> rng;
    std::ostringstream line;
    line << "LUMIRANDOM " << lumi.run() << " " << lumi.luminosityBlock() << " " << moduleDescription().moduleLabel()
         << draws(rng->getEngine(lumi.index()), draws_) << "\n";
    std::cout << line.str() << std::flush;
    return nullptr;
  }

  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {}

private:
  unsigned int const draws_;
};

DEFINE_FWK_MODULE(TestEventSeededRandom);
