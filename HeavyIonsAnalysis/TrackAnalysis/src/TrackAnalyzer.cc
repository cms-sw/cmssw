#include <algorithm>
#include <utility>
#include "FWCore/Utilities/interface/Exception.h"

#include "HeavyIonsAnalysis/TrackAnalysis/interface/TrackAnalyzer.h"

TrackAnalyzer::TrackAnalyzer(const edm::ParameterSet& iConfig) :
  doTrack_(iConfig.getUntrackedParameter<bool>("doTrack", true)),
  trackPtMin_(iConfig.getUntrackedParameter<double>("trackPtMin", 0.01)),
  trackEtaMax_(iConfig.getUntrackedParameter<double>("trackEtaMax", 4.0)),
  applyTrackSelections_(iConfig.getUntrackedParameter<bool>("applyTrackSelections", false)),
  branchesToKeep_(iConfig.getUntrackedParameter<std::vector<std::string>>("branchesToKeep", std::vector<std::string>{})),
  vertexSrc_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexSrc"))),
  trackSrc_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trackSrc"))),
  track2pcSrc_(consumes<std::vector<edm::Ptr<pat::PackedCandidate> > >(iConfig.getParameter<edm::InputTag>("trackSrc"))),
  beamSpotProducer_(consumes<reco::BeamSpot>(
      iConfig.getUntrackedParameter<edm::InputTag>("beamSpotSrc", edm::InputTag("offlineBeamSpot")))) {
  for (const auto& tag : iConfig.getParameter<std::vector<edm::InputTag>>("dedxEstimators")) {
    const auto label = tag.instance()!="" ? tag.instance() : tag.label();
    dedxEstimatorsSrc_.emplace(label, consumes<edm::ValueMap<reco::DeDxData> >(tag));
  }
}

//--------------------------------------------------------------------------------------------------
TrackAnalyzer::~TrackAnalyzer() {}

//--------------------------------------------------------------------------------------------------
void TrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  nEv = (int)iEvent.id().event();
  nRun = (int)iEvent.id().run();
  nLumi = (int)iEvent.luminosityBlock();

  clearVectors();

  fillVertices(iEvent);

  if (doTrack_)
    fillTracks(iEvent, iSetup);

  trackTree_->Fill();
}

//--------------------------------------------------------------------------------------------------
void TrackAnalyzer::fillVertices(const edm::Event& iEvent) {
  // Fill reconstructed vertices.
  const auto& recoVertices = iEvent.get(vertexSrc_);

  iMaxPtSumVtx = -1;
  float maxPtSum = -999;
  nVtx = (int)recoVertices.size();
  for (int i = 0; i < nVtx; ++i) {
    const auto& v = recoVertices[i];
    xVtx.push_back(v.position().x());
    yVtx.push_back(v.position().y());
    zVtx.push_back(v.position().z());
    xErrVtx.push_back(v.xError());
    yErrVtx.push_back(v.yError());
    zErrVtx.push_back(v.zError());

    chi2Vtx.push_back(v.chi2());
    ndofVtx.push_back(v.ndof());

    isFakeVtx.push_back(v.isFake());

    //number of tracks having a weight in vtx fit above 0.5
    nTracksVtx.push_back(v.nTracks());

    float ptSum = 0;
    for (const auto& track : v.tracks())
      ptSum += track->pt();

    ptSumVtx.push_back(ptSum);
    if (ptSum > maxPtSum) {
      iMaxPtSumVtx = i;
      maxPtSum = ptSum;
    }
  }
}

//--------------------------------------------------------------------------------------------------
void TrackAnalyzer::fillTracks(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& tracks = iEvent.getHandle(trackSrc_);
  const auto& track2pc = iEvent.getHandle(track2pcSrc_);
  std::map<std::string, edm::ValueMap<reco::DeDxData> > dedxMaps;
  for (const auto& d : dedxEstimatorsSrc_)
    dedxMaps.emplace(d.first, iEvent.get(d.second));

  //loop over tracks
  for (unsigned it = 0; it < tracks->size(); ++it) {
    const auto& t = tracks->at(it);
    const auto& k = track2pc.isValid() ? track2pc->at(it) : edm::Ptr<pat::PackedCandidate>();
    const auto& c = k.isNonnull() ? *k : pat::PackedCandidate();

    if (t.pt() < trackPtMin_)
      continue;

    if (fabs(t.eta()) > trackEtaMax_)
	    continue;

    if (applyTrackSelections_){
      if (t.quality(reco::TrackBase::qualityByName("highPurity")) == false) // only high-purity tracks
	      continue;
      if (t.ptError() / t.pt() > 0.1) // only tracks with pT resolution better than 10%
	      continue;
    }

    trkPt.push_back(t.pt());
    trkPtError.push_back(t.ptError());
    trkEta.push_back(t.eta());
    trkPhi.push_back(t.phi());
    trkCharge.push_back((char)t.charge());
    trkPDGId.push_back(c.pdgId());
    trkNHits.push_back((char)t.numberOfValidHits());
    trkNPixHits.push_back((char)t.hitPattern().numberOfValidPixelHits());
    trkNLayers.push_back((char)t.hitPattern().trackerLayersWithMeasurement());
    highPurity.push_back(t.quality(reco::TrackBase::qualityByName("highPurity")));
    trkNormChi2.push_back(t.normalizedChi2());

    pfEnergy.push_back(c.energy());
    pfEcal.push_back(c.energy() * (c.caloFraction() - c.hcalFraction()));
    pfHcal.push_back(c.energy() * c.hcalFraction());

    //DCA info for associated vtx
    const auto& v = c.vertexRef();
    if (v.isNonnull() && c.hasTrackDetails()) {
      trkAssociatedVtxIndx.push_back(v.key());
      trkAssociatedVtxQuality.push_back(c.fromPV(v.key()));
      trkDzAssociatedVtx.push_back(c.dz(v->position()));
      trkDzErrAssociatedVtx.push_back(sqrt(c.dzError() * c.dzError() + v->zError() * v->zError()));
      trkDxyAssociatedVtx.push_back(c.dxy(v->position()));
      trkDxyErrAssociatedVtx.push_back(sqrt(c.dxyError() * c.dxyError() + v->xError() * v->yError()));
    } else {
      trkAssociatedVtxIndx.push_back(-1);
      trkAssociatedVtxQuality.push_back(-999999);
      trkDzAssociatedVtx.push_back(-999999);
      trkDzErrAssociatedVtx.push_back(-999999);
      trkDxyAssociatedVtx.push_back(-999999);
      trkDxyErrAssociatedVtx.push_back(-999999);
    }

    //DCA info for first (highest pt) vtx
    if (iMaxPtSumVtx >= 0) {
      math::XYZPoint v(xVtx.at(iMaxPtSumVtx), yVtx.at(iMaxPtSumVtx), zVtx.at(iMaxPtSumVtx));
      trkFirstVtxQuality.push_back(c.fromPV(iMaxPtSumVtx));
      trkDzFirstVtx.push_back(t.dz(v));
      // WARNING !! reco::Track::dzError() and pat::PackedCandidate::dzError() give different values. Former must be used for HIN track   ID.
      trkDzErrFirstVtx.push_back(sqrt(t.dzError() * t.dzError() + zErrVtx.at(iMaxPtSumVtx) * zErrVtx.at(iMaxPtSumVtx)));
      trkDxyFirstVtx.push_back(t.dxy(v));
      trkDxyErrFirstVtx.push_back(sqrt(t.dxyError() * t.dxyError() + xErrVtx.at(iMaxPtSumVtx) * yErrVtx.at(iMaxPtSumVtx)));
    } else {
      trkFirstVtxQuality.push_back(-999999);
      trkDzFirstVtx.push_back(-999999);
      trkDzErrFirstVtx.push_back(-999999);
      trkDxyFirstVtx.push_back(-999999);
      trkDxyErrFirstVtx.push_back(-999999);
    }
    for (auto& d : trkDeDx) {
      double dEdx(-99.9);
      if (track2pc.isValid() && dedxMaps.at(d.first).contains(k.id()))
        dEdx = dedxMaps.at(d.first)[k].dEdx();
      else if (dedxMaps.at(d.first).contains(tracks.id()))
        dEdx = dedxMaps.at(d.first)[reco::TrackRef(tracks, it)].dEdx();
      d.second.push_back(dEdx);
    }

    nTrk++;
  }
}

// ------------ method called once each job just before starting event loop  ------------
void TrackAnalyzer::beginJob() {
  trackTree_ = fs->make<TTree>("trackTree", "v1");

  // Filter branch booking, not branch reading: omitted payloads are never written.
  const auto bookBranch = [this](const char* name, auto&&... args) {
    if (branchesToKeep_.empty() ||
        std::find(branchesToKeep_.begin(), branchesToKeep_.end(), name) != branchesToKeep_.end())
      trackTree_->Branch(name, std::forward<decltype(args)>(args)...);
  };

  // event
  bookBranch("nRun", &nRun, "nRun/I");
  bookBranch("nEv", &nEv, "nEv/I");
  bookBranch("nLumi", &nLumi, "nLumi/I");

  // vertex
  bookBranch("nVtx", &nVtx);
  bookBranch("xVtx", &xVtx);
  bookBranch("yVtx", &yVtx);
  bookBranch("zVtx", &zVtx);
  bookBranch("xErrVtx", &xErrVtx);
  bookBranch("yErrVtx", &yErrVtx);
  bookBranch("zErrVtx", &zErrVtx);
  bookBranch("chi2Vtx", &chi2Vtx);
  bookBranch("ndofVtx", &ndofVtx);
  bookBranch("isFakeVtx", &isFakeVtx);
  bookBranch("nTracksVtx", &nTracksVtx);
  bookBranch("ptSumVtx", &ptSumVtx);

  // Tracks
  bookBranch("nTrk", &nTrk);
  bookBranch("trkPt", &trkPt);
  bookBranch("trkPtError", &trkPtError);
  bookBranch("trkEta", &trkEta);
  bookBranch("trkPhi", &trkPhi);
  bookBranch("trkCharge", &trkCharge);
  bookBranch("trkPDGId", &trkPDGId);
  bookBranch("trkNHits", &trkNHits);
  bookBranch("trkNPixHits", &trkNPixHits);
  bookBranch("trkNLayers", &trkNLayers);
  bookBranch("trkNormChi2", &trkNormChi2);
  bookBranch("highPurity", &highPurity);

  bookBranch("pfEnergy", &pfEnergy);
  bookBranch("pfEcal", &pfEcal);
  bookBranch("pfHcal", &pfHcal);

  bookBranch("trkAssociatedVtxIndx", &trkAssociatedVtxIndx);
  bookBranch("trkAssociatedVtxQuality", &trkAssociatedVtxQuality);
  bookBranch("trkDzAssociatedVtx", &trkDzAssociatedVtx);
  bookBranch("trkDzErrAssociatedVtx", &trkDzErrAssociatedVtx);
  bookBranch("trkDxyAssociatedVtx", &trkDxyAssociatedVtx);
  bookBranch("trkDxyErrAssociatedVtx", &trkDxyErrAssociatedVtx);
  bookBranch("trkFirstVtxQuality", &trkFirstVtxQuality);
  bookBranch("trkDzFirstVtx", &trkDzFirstVtx);
  bookBranch("trkDzErrFirstVtx", &trkDzErrFirstVtx);
  bookBranch("trkDxyFirstVtx", &trkDxyFirstVtx);
  bookBranch("trkDxyErrFirstVtx", &trkDxyErrFirstVtx);
  for (const auto& d : dedxEstimatorsSrc_)
    bookBranch(d.first.c_str(), &(trkDeDx[d.first]));

  for (const auto& name : branchesToKeep_)
    if (!trackTree_->GetBranch(name.c_str()))
      throw cms::Exception("Configuration") << "branchesToKeep requested an unavailable branch: " << name;
}

// ------------ method called once each job just after ending the event loop  ------------
void TrackAnalyzer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackAnalyzer);
