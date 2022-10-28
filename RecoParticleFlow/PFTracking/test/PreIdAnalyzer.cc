#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/ParticleFlowReco/interface/PreIdFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "DataFormats/Common/interface/ValueMap.h"

class PreIdAnalyzer : public DQMEDAnalyzer {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit PreIdAnalyzer(const edm::ParameterSet&);
  ~PreIdAnalyzer() override = default;

  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override {}
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  //  virtual void beginJobAnalyze(const edm::EventSetup & c);
private:
  const edm::EDGetTokenT<edm::ValueMap<reco::PreIdRef> > preIdMapToken_;
  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;

  MonitorElement* tracksPt_;
  MonitorElement* tracksEta_;
  MonitorElement* tracksPtEcalMatch_;
  MonitorElement* tracksEtaEcalMatch_;
  MonitorElement* geomMatchChi2_;
  MonitorElement* geomMatchEop_;

  MonitorElement* tracksChi2_;
  MonitorElement* tracksNhits_;
  MonitorElement* tracksPtFiltered_;
  MonitorElement* tracksEtaFiltered_;
  MonitorElement* tracksPtNotFiltered_;
  MonitorElement* tracksEtaNotFiltered_;

  MonitorElement* tracksPtPreIded_;
  MonitorElement* tracksEtaPreIded_;
  MonitorElement* trackdpt_;
  MonitorElement* gsfChi2_;
  MonitorElement* chi2Ratio_;
  MonitorElement* mva_;
};

PreIdAnalyzer::PreIdAnalyzer(const edm::ParameterSet& pset)
    : preIdMapToken_(consumes<edm::ValueMap<reco::PreIdRef> >(pset.getParameter<edm::InputTag>("PreIdMap"))),
      trackToken_(consumes<reco::TrackCollection>(pset.getParameter<edm::InputTag>("TrackCollection"))) {}

void PreIdAnalyzer::bookHistograms(DQMStore::IBooker& dbe, edm::Run const&, edm::EventSetup const&) {
  //void  PreIdAnalyzer::beginJobAnalyze(const edm::EventSetup & c){
  tracksPt_ = dbe.book1D("TracksPt", "pT", 1000, 0, 100.);
  tracksEta_ = dbe.book1D("TracksEta", "eta", 50, -2.5, 2.5);
  tracksPtEcalMatch_ = dbe.book1D("TracksPtEcalMatch", "pT", 1000, 0, 100.);
  tracksEtaEcalMatch_ = dbe.book1D("TracksEtaEcalMatch", "eta", 50, -2.5, 2.5);
  tracksPtFiltered_ = dbe.book1D("TracksPtFiltered", "pT", 1000, 0, 100.);
  tracksEtaFiltered_ = dbe.book1D("TracksEtaFiltered", "eta", 50, -2.5, 2.5);
  tracksPtNotFiltered_ = dbe.book1D("TracksPtNotFiltered", "pT", 1000, 0, 100.);
  tracksEtaNotFiltered_ = dbe.book1D("TracksEtaNotFiltered", "eta", 50, -2.5, 2.5);
  tracksPtPreIded_ = dbe.book1D("TracksPtPreIded", "pT", 1000, 0, 100.);
  tracksEtaPreIded_ = dbe.book1D("TracksEtaPreIded", "eta", 50, -2.5, 2.5);
  tracksChi2_ = dbe.book1D("TracksChi2", "chi2", 100, 0, 10.);
  tracksNhits_ = dbe.book1D("TracksNhits", "Nhits", 30, -0.5, 29.5);

  geomMatchChi2_ = dbe.book1D("geomMatchChi2", "Geom Chi2", 100, 0., 50.);
  geomMatchEop_ = dbe.book1D("geomMatchEop", "E/p", 100, 0., 5.);
  trackdpt_ = dbe.book1D("trackdpt", "dpt/pt", 100, 0., 5.);
  gsfChi2_ = dbe.book1D("gsfChi2", "GSF chi2", 100, 0., 10.);
  chi2Ratio_ = dbe.book1D("chi2Ratio", "Chi2 ratio", 100, 0., 10.);
  mva_ = dbe.book1D("mva", "mva", 100, -1., 1.);
}

void PreIdAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& trackh = iEvent.getHandle(trackToken_);
  auto const& vmaph = iEvent.getHandle(preIdMapToken_);

  const reco::TrackCollection& tracks = *(trackh.product());
  const edm::ValueMap<reco::PreIdRef>& preidMap = *(vmaph.product());

  unsigned ntracks = tracks.size();
  for (unsigned itrack = 0; itrack < ntracks; ++itrack) {
    reco::TrackRef theTrackRef(trackh, itrack);
    tracksPt_->Fill(theTrackRef->pt());
    tracksEta_->Fill(theTrackRef->eta());

    if (preidMap[theTrackRef].isNull())
      continue;

    const reco::PreId& myPreId(*(preidMap[theTrackRef]));
    geomMatchChi2_->Fill(myPreId.geomMatching()[4]);
    geomMatchEop_->Fill(myPreId.eopMatch());

    if (myPreId.ecalMatching()) {
      tracksPtEcalMatch_->Fill(theTrackRef->pt());
      tracksEtaEcalMatch_->Fill(theTrackRef->eta());
    } else {
      tracksChi2_->Fill(myPreId.kfChi2());
      tracksNhits_->Fill(myPreId.kfNHits());
      if (myPreId.trackFiltered()) {
        tracksPtFiltered_->Fill(theTrackRef->pt());
        tracksEtaFiltered_->Fill(theTrackRef->eta());
        trackdpt_->Fill(myPreId.dpt());
        gsfChi2_->Fill(myPreId.gsfChi2());
        chi2Ratio_->Fill(myPreId.chi2Ratio());
        mva_->Fill(myPreId.mva());
      } else {
        tracksPtNotFiltered_->Fill(theTrackRef->pt());
        tracksEtaNotFiltered_->Fill(theTrackRef->eta());
      }
    }
    if (myPreId.preIded()) {
      tracksPtPreIded_->Fill(theTrackRef->pt());
      tracksEtaPreIded_->Fill(theTrackRef->eta());
    }
  }
}

DEFINE_FWK_MODULE(PreIdAnalyzer);
