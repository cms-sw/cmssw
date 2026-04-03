// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/transform.h"  // for edm::vector_transform

// ROOT includes
#include "TLorentzVector.h"

// standard includes
#include <fmt/printf.h>

class ShortenedTrackResolution : public DQMEDAnalyzer {
public:
  ShortenedTrackResolution(const edm::ParameterSet &);
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  const std::string folderName_;
  const std::vector<std::string> hitsRemain_;
  const double minTracksEta_;
  const double maxTracksEta_;
  const double minTracksPt_;
  const double maxTracksPt_;

  const double maxDr_;
  const edm::InputTag tracksTag_;
  const std::vector<edm::InputTag> tracksRerecoTag_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracksToken_;
  const std::vector<edm::EDGetTokenT<std::vector<reco::Track>>> tracksRerecoToken_;
  const edm::InputTag beamSpotTag_;
  const edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;

  // resolutions
  std::vector<MonitorElement *> histsPtRatioAll_;
  std::vector<MonitorElement *> histsPtDiffAll_;
  std::vector<MonitorElement *> histsEtaDiffAll_;
  std::vector<MonitorElement *> histsPhiDiffAll_;
  std::vector<MonitorElement *> histsdxyDiffAll_;
  std::vector<MonitorElement *> histsdzDiffAll_;
  std::vector<MonitorElement *> histsPtRatioVsDeltaRAll_;
  std::vector<MonitorElement *> histsDeltaPtOverPtAll_;

  // properties
  std::vector<MonitorElement *> histsPtAll_;
  std::vector<MonitorElement *> histsNhitsAll_;
  std::vector<MonitorElement *> histsDeltaRAll_;
  std::vector<MonitorElement *> histsdxyAll_;
  std::vector<MonitorElement *> histsdzAll_;
};

// -----------------------------
// constructors and destructor
// -----------------------------
ShortenedTrackResolution::ShortenedTrackResolution(const edm::ParameterSet &ps)
    : folderName_(ps.getUntrackedParameter<std::string>("folderName", "TrackRefitting")),
      hitsRemain_(ps.getUntrackedParameter<std::vector<std::string>>("hitsRemainInput")),
      minTracksEta_(ps.getUntrackedParameter<double>("minTracksEtaInput", 0.0)),
      maxTracksEta_(ps.getUntrackedParameter<double>("maxTracksEtaInput", 2.2)),
      minTracksPt_(ps.getUntrackedParameter<double>("minTracksPtInput", 15.0)),
      maxTracksPt_(ps.getUntrackedParameter<double>("maxTracksPtInput", 99999.9)),
      maxDr_(ps.getUntrackedParameter<double>("maxDrInput", 0.01)),
      tracksTag_(ps.getUntrackedParameter<edm::InputTag>("tracksInputTag", edm::InputTag("generalTracks", "", "DQM"))),
      tracksRerecoTag_(ps.getUntrackedParameter<std::vector<edm::InputTag>>("tracksRerecoInputTag")),
      tracksToken_(consumes<std::vector<reco::Track>>(tracksTag_)),
      tracksRerecoToken_(edm::vector_transform(
          tracksRerecoTag_, [this](edm::InputTag const &tag) { return consumes<std::vector<reco::Track>>(tag); })),
      beamSpotTag_(ps.getUntrackedParameter<edm::InputTag>("BeamSpotTag", edm::InputTag("offlineBeamSpot"))),
      beamspotToken_(consumes<reco::BeamSpot>(beamSpotTag_)) {
  const size_t n = hitsRemain_.size();
  histsPtRatioAll_.reserve(n);
  histsPtDiffAll_.reserve(n);
  histsEtaDiffAll_.reserve(n);
  histsPhiDiffAll_.reserve(n);
  histsdxyDiffAll_.reserve(n);
  histsdzDiffAll_.reserve(n);
  histsPtRatioVsDeltaRAll_.reserve(n);
  histsDeltaPtOverPtAll_.reserve(n);
  histsPtAll_.reserve(n);
  histsNhitsAll_.reserve(n);
  histsDeltaRAll_.reserve(n);
  histsdxyAll_.reserve(n);
  histsdzAll_.reserve(n);
}

//__________________________________________________________________________________
void ShortenedTrackResolution::bookHistograms(DQMStore::IBooker &iBook,
                                              edm::Run const &iRun,
                                              edm::EventSetup const &iSetup) {
  auto book1D = [&](const std::string &name, const std::string &title, int bins, double min, double max) {
    return iBook.book1D(name.c_str(), title.c_str(), bins, min, max);
  };

  auto book2D = [&](const std::string &name,
                    const std::string &title,
                    int binsX,
                    double minX,
                    double maxX,
                    int binsY,
                    double minY,
                    double maxY) {
    return iBook.book2D(name.c_str(), title.c_str(), binsX, minX, maxX, binsY, minY, maxY);
  };

  std::string currentFolder = folderName_ + "/Resolutions";
  iBook.setCurrentFolder(currentFolder);

  for (const auto &label : hitsRemain_) {
    std::string name, title;

    name = fmt::sprintf("trackPtRatio_%s", label);
    title =
        fmt::sprintf("Short Track p_{T} / Full Track p_{T} - %s layers;p_{T}^{short}/p_{T}^{full};n. tracks", label);
    histsPtRatioAll_.push_back(book1D(name, title, 100, 0.5, 1.5));

    name = fmt::sprintf("trackPtDiff_%s", label);
    title = fmt::sprintf(
        "Short Track p_{T} - Full Track p_{T} - %s layers;p_{T}^{short} - p_{T}^{full} [GeV];n. tracks", label);
    histsPtDiffAll_.push_back(book1D(name, title, 100, -10., 10.));

    name = fmt::sprintf("trackEtaDiff_%s", label);
    title = fmt::sprintf("Short Track #eta - Full Track #eta - %s layers;#eta^{short} - #eta^{full};n. tracks", label);
    histsEtaDiffAll_.push_back(book1D(name, title, 100, -0.001, 0.001));

    name = fmt::sprintf("trackPhiDiff_%s", label);
    title = fmt::sprintf("Short Track #phi - Full Track #phi - %s layers;#phi^{short} - #phi^{full};n. tracks", label);
    histsPhiDiffAll_.push_back(book1D(name, title, 100, -0.001, 0.001));

    name = fmt::sprintf("trackdxyDiff_%s", label);
    title = fmt::sprintf(
        "Short Track d_{xy} - Full Track d_{xy} - %s layers;d_{xy}^{short} - d_{xy}^{full} [cm];n. tracks", label);
    histsdxyDiffAll_.push_back(book1D(name, title, 100, -0.1, 0.1));

    name = fmt::sprintf("trackdzDiff_%s", label);
    title = fmt::sprintf("Short Track d_{z} - Full Track d_{z} - %s layers;d_{z}^{short} - d_{z}^{full} [cm];n. tracks",
                         label);
    histsdzDiffAll_.push_back(book1D(name, title, 100, -0.1, 0.1));

    name = fmt::sprintf("trackPtRatioVsDeltaR_%s", label);
    title = fmt::sprintf(
        "Short Track p_{T} / Full Track p_{T} - %s layers vs "
        "#DeltaR;#DeltaR(short,full);p_{T}^{short}/p_{T}^{full} [GeV];n. tracks",
        label);
    histsPtRatioVsDeltaRAll_.push_back(book2D(name, title, 100, 0., 0.01, 101, -0.05, 2.05));

    name = fmt::sprintf("trackDeltaPtOverPt_%s", label);
    title = fmt::sprintf(
        "Short Track p_{T} - Full Track p_{T} / Full Track p_{T} - %s layers;"
        "p_{T}^{short} - p_{T}^{full} / p^{full}_{T};n. tracks",
        label);
    histsDeltaPtOverPtAll_.push_back(book1D(name, title, 101, -5., 5.));
  }

  currentFolder = folderName_ + "/TrackProperties";
  iBook.setCurrentFolder(currentFolder);

  for (const auto &label : hitsRemain_) {
    std::string name, title;

    name = fmt::sprintf("trackPt_%s", label);
    title = fmt::sprintf("Short Track p_{T} - %s layers;p_{T}^{short} [GeV];n. tracks", label);
    histsPtAll_.push_back(book1D(name, title, 100, 0., 100.));

    name = fmt::sprintf("trackNhits_%s", label);
    title = fmt::sprintf("Short Track n. hits - %s layers;n. hits per track;n. tracks", label);
    histsNhitsAll_.push_back(book1D(name, title, 20, -0.5, 19.5));

    name = fmt::sprintf("trackDxy_%s", label);
    title = fmt::sprintf("Short Track d_{xy} - %s layers;track d_{xy} [cm];n. tracks", label);
    histsdxyAll_.push_back(book1D(name, title, 100, -0.025, 0.025));

    name = fmt::sprintf("trackDz_%s", label);
    title = fmt::sprintf("Short Track d_{z} - %s layers;track d_{z} [cm];n. tracks", label);
    histsdzAll_.push_back(book1D(name, title, 80, -20, 20));

    name = fmt::sprintf("trackDeltaR_%s", label);
    title = fmt::sprintf("Short Track / Full Track #DeltaR - %s layers;#DeltaR(short,full);n. tracks", label);
    histsDeltaRAll_.push_back(book1D(name, title, 100, 0., 0.005));
  }
}

//__________________________________________________________________________________
void ShortenedTrackResolution::analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup) {
  const auto &tracks = iEvent.getHandle(tracksToken_);

  if (!tracks.isValid()) {
    edm::LogError("ShortenedTrackResolution") << "Missing input track collection " << tracksTag_.encode() << std::endl;
    return;
  }

  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> beamSpotHandle = iEvent.getHandle(beamspotToken_);
  if (beamSpotHandle.isValid()) {
    beamSpot = *beamSpotHandle;
  } else {
    beamSpot = reco::BeamSpot();
  }

  math::XYZPoint BS(beamSpot.x0(), beamSpot.y0(), beamSpot.z0());

  for (const auto &track : *tracks) {
    const reco::HitPattern &hp = track.hitPattern();
    if (int(int(hp.numberOfValidHits()) - int(hp.numberOfAllHits(reco::HitPattern::TRACK_HITS))) != 0) {
      break;
    }

    TLorentzVector tvec;
    tvec.SetPtEtaPhiM(track.pt(), track.eta(), track.phi(), 0.0);

    int i = 0;  // token index
    for (const auto &token : tracksRerecoToken_) {
      const auto &tracks_rereco = iEvent.getHandle(token);

      for (const auto &track_rereco : *tracks_rereco) {
        TLorentzVector trerecovec;
        trerecovec.SetPtEtaPhiM(track_rereco.pt(), track_rereco.eta(), track_rereco.phi(), 0.0);
        double deltaR = tvec.DeltaR(trerecovec);

        if (deltaR < maxDr_) {
          if (track_rereco.pt() >= minTracksPt_ && track_rereco.pt() <= maxTracksPt_ &&
              std::abs(track_rereco.eta()) >= minTracksEta_ && std::abs(track_rereco.eta()) <= maxTracksEta_) {
            // resolutions
            histsPtRatioAll_[i]->Fill(1.0 * track_rereco.pt() / track.pt());
            histsPtDiffAll_[i]->Fill(track_rereco.pt() - track.pt());
            histsDeltaPtOverPtAll_[i]->Fill((track_rereco.pt() - track.pt()) / track.pt());
            histsEtaDiffAll_[i]->Fill(track_rereco.eta() - track.eta());
            histsPhiDiffAll_[i]->Fill(track_rereco.phi() - track.phi());
            histsdxyDiffAll_[i]->Fill(track_rereco.dxy(BS) - track.dxy(BS));
            histsdzDiffAll_[i]->Fill(track_rereco.dz(BS) - track.dz(BS));
            histsPtRatioVsDeltaRAll_[i]->Fill(deltaR, track_rereco.pt() / track.pt());

            // properties
            histsPtAll_[i]->Fill(track_rereco.pt());
            histsdxyAll_[i]->Fill(track.dxy(BS));
            histsdzAll_[i]->Fill(track.dz(BS));
            histsNhitsAll_[i]->Fill(track_rereco.numberOfValidHits());
            histsDeltaRAll_[i]->Fill(deltaR);
          }
        }
      }
      ++i;
    }
  }
}

//__________________________________________________________________________________
void ShortenedTrackResolution::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("folderName", "TrackRefitting");
  desc.addUntracked<std::vector<std::string>>("hitsRemainInput", {});
  desc.addUntracked<double>("minTracksEtaInput", 0.0);
  desc.addUntracked<double>("maxTracksEtaInput", 2.2);
  desc.addUntracked<double>("minTracksPtInput", 15.0);
  desc.addUntracked<double>("maxTracksPtInput", 99999.9);
  desc.addUntracked<double>("maxDrInput", 0.01);
  desc.addUntracked<edm::InputTag>("tracksInputTag", edm::InputTag("generalTracks", "", "DQM"));
  desc.addUntracked<std::vector<edm::InputTag>>("tracksRerecoInputTag", {});
  descriptions.addWithDefaultLabel(desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ShortenedTrackResolution);
