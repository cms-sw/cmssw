#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"

#include <memory>

namespace trackwordtest {
  class TTTrackTrackWordDummyOneAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit TTTrackTrackWordDummyOneAnalyzer(const edm::ParameterSet&) {}
    ~TTTrackTrackWordDummyOneAnalyzer() override {}

    void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override {
      TTTrack_TrackWord tw;
      tw.testDigitizationScheme();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      //to ensure distinct cfi names
      descriptions.addWithDefaultLabel(desc);
    }
  };
}  // namespace trackwordtest

using trackwordtest::TTTrackTrackWordDummyOneAnalyzer;
DEFINE_FWK_MODULE(TTTrackTrackWordDummyOneAnalyzer);
