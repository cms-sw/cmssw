// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
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

  std::vector<MonitorElement *> histsPtAll_;
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
          tracksRerecoTag_, [this](edm::InputTag const &tag) { return consumes<std::vector<reco::Track>>(tag); })) {
  histsPtAll_.clear();
}

//__________________________________________________________________________________
void ShortenedTrackResolution::bookHistograms(DQMStore::IBooker &iBook,
                                              edm::Run const &iRun,
                                              edm::EventSetup const &iSetup) {
  std::string currentFolder = folderName_ + "/";
  iBook.setCurrentFolder(currentFolder);

  for (int i = 0; i < int(hitsRemain_.size()); ++i) {
    histsPtAll_.push_back(iBook.book1D(
        fmt::sprintf("trackPtRatio_%s", hitsRemain_[i]).c_str(),
        fmt::sprintf("Short Track p_{T} / Full Track p_{T} - %s layers;p_{T}^{short}/p_{T}^{full};n. tracks",
                     hitsRemain_[i])
            .c_str(),
        101,
        -0.05,
        2.05));
  }
}

//__________________________________________________________________________________
void ShortenedTrackResolution::analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup) {
  const auto &tracks = iEvent.getHandle(tracksToken_);

  if (!tracks.isValid()) {
    edm::LogError("ShortenedTrackResolution") << "Missing input track collection " << tracksTag_.encode() << std::endl;
    return;
  }

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
            histsPtAll_[i]->Fill(1.0 * track_rereco.pt() / track.pt());
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
