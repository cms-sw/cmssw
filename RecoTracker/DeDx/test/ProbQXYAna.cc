// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"

#include "DataFormats/TrackReco/interface/SiPixelTrackProbQXY.h"

class ProbQXYAna : public edm::global::EDAnalyzer<> {
public:
  explicit ProbQXYAna(const edm::ParameterSet&);
  ~ProbQXYAna() override = default;

private:
  virtual void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<std::vector<pat::IsolatedTrack>> trackToken_;
};

using namespace reco;
using namespace std;
using namespace edm;

ProbQXYAna::ProbQXYAna(const edm::ParameterSet& iConfig)
    : trackToken_(consumes<std::vector<pat::IsolatedTrack>>(iConfig.getParameter<edm::InputTag>("tracks"))) {}

void ProbQXYAna::analyze(edm::StreamID id, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<std::vector<pat::IsolatedTrack>> trackCollectionHandle;
  iEvent.getByToken(trackToken_, trackCollectionHandle);
  const auto trackCollection(*trackCollectionHandle.product());
  int numTrack = 0;
  for (const auto& track : trackCollection) {
    numTrack++;
    float probQonTrack = track.probQonTrack();
    float probXYonTrack = track.probXYonTrack();
    float probQonTrackNoLayer1 = track.probQonTrackNoLayer1();
    float probXYonTrackNoLayer1 = track.probXYonTrackNoLayer1();
    LogPrint("ProbQXYAna") << "--------------------------------------------------";
    LogPrint("ProbQXYAna") << "For track " << numTrack;
    LogPrint("ProbQXYAna") << "probQonTrack: " << probQonTrack << " and probXYonTrack: " << probXYonTrack;
    LogPrint("ProbQXYAna") << "probQonTrackNoLayer1: " << probQonTrackNoLayer1
                           << " and probXYonTrackNoLayer1: " << probXYonTrackNoLayer1;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ProbQXYAna);
