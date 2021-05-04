#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ConversionFinder.h"

namespace egamma::conv {

  std::ostream& operator<<(std::ostream& os, ConversionInfo const& conv) {
    auto partnerIdx = conv.conversionPartnerCtfTkIdx.value_or(conv.conversionPartnerGsfTkIdx.value_or(-9999));

    return os << "  - flag: " << conv.flag << "\n  - partnerIdx:" << partnerIdx << "\n  - dist: " << conv.dist
              << "\n  - dcot: " << conv.dcot << "\n  - radius: " << conv.radiusOfConversion
              << "\n  - deltaMissingHits: " << conv.deltaMissingHits;
  }

}  // namespace egamma::conv

class TestGsfElectronConversionFinder : public edm::one::EDAnalyzer<> {
public:
  explicit TestGsfElectronConversionFinder(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<edm::View<reco::GsfElectron>> gsfElectronsToken_;
};

TestGsfElectronConversionFinder::TestGsfElectronConversionFinder(const edm::ParameterSet& pset)
    : gsfElectronsToken_{consumes<edm::View<reco::GsfElectron>>(pset.getParameter<edm::InputTag>("gsfElectrons"))} {}

void TestGsfElectronConversionFinder::analyze(const edm::Event& event, const edm::EventSetup&) {
  // this is just a test, so we can hardcode some magnetic field value
  constexpr float magneticFieldInTesla = 4.0f;

  // get electron collection
  auto const& gsfElectrons = event.get(gsfElectronsToken_);

  // conversion finder requires track tables
  std::optional<egamma::conv::TrackTable> ctfTrackTable = std::nullopt;
  std::optional<egamma::conv::TrackTable> gsfTrackTable = std::nullopt;

  int gsfElectronIdx{0};

  // get the original CTF and GSF tracks
  for (auto const& gsfElectron : gsfElectrons) {
    auto const& gsfTrack = gsfElectron.core()->gsfTrack();
    auto const& ctfTrack = gsfElectron.core()->ctfTrack();
    edm::LogVerbatim("TestConversionFinder") << "===================================" << std::endl;
    edm::LogVerbatim("TestConversionFinder")
        << "event " << event.id().event() << ", gsfElectron " << gsfElectronIdx << std::endl;
    edm::LogVerbatim("TestConversionFinder") << "-------------------------" << std::endl;
    if (gsfTrackTable == std::nullopt) {
      edm::Handle<reco::GsfTrackCollection> originalGsfTracksHandle;
      event.get(gsfTrack.id(), originalGsfTracksHandle);
      gsfTrackTable = egamma::conv::TrackTable(*originalGsfTracksHandle);
    }
    if (ctfTrackTable == std::nullopt) {
      if (!ctfTrack.isNull()) {
        edm::Handle<reco::TrackCollection> originalCtfTracksHandle;
        event.get(ctfTrack.id(), originalCtfTracksHandle);
        ctfTrackTable = egamma::conv::TrackTable(*originalCtfTracksHandle);
      }
    }

    // run the converion finder
    // Taking an empty CTF track table if it was not possible to obtain the
    // track collection from the electron is a hack for testing purposes.
    auto conversions = egamma::conv::findConversions(*gsfElectron.core(),
                                                     ctfTrackTable.value_or(egamma::conv::TrackTable{}),
                                                     gsfTrackTable.value(),
                                                     magneticFieldInTesla,
                                                     0.45f);
    auto bestConversion = egamma::conv::findBestConversionMatch(conversions);

    // log conversion info
    int conversionIdx{0};
    for (auto const& conversion : conversions) {
      edm::LogVerbatim("TestConversionFinder") << "conversion " << conversionIdx << std::endl;
      edm::LogVerbatim("TestConversionFinder") << conversion << std::endl;
      edm::LogVerbatim("TestConversionFinder") << "-------------------------" << std::endl;
      ++conversionIdx;
    }
    edm::LogVerbatim("TestConversionFinder") << "best conversion " << std::endl;
    edm::LogVerbatim("TestConversionFinder") << bestConversion << std::endl;

    ++gsfElectronIdx;
  }
}

void TestGsfElectronConversionFinder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gsfElectrons", {"gedGsfElectrons"});
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestGsfElectronConversionFinder);
