#include "HeavyIonsAnalysis/TrackAnalysis/interface/TrackAnalyzer.h"

TrackAnalyzer::TrackAnalyzer(const edm::ParameterSet& iConfig) :
  doTrack_(iConfig.getUntrackedParameter<bool>("doTrack", true)),
  trackPtMin_(iConfig.getUntrackedParameter<double>("trackPtMin", 0.01)),
  trackEtaMax_(iConfig.getUntrackedParameter<double>("trackEtaMax", 4.0)),
  applyTrackSelections_(iConfig.getUntrackedParameter<bool>("applyTrackSelections", false)),
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

    if (std::abs(t.eta()) > trackEtaMax_)
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

  // event
  trackTree_->Branch("nRun", &nRun, "nRun/I");
  trackTree_->Branch("nEv", &nEv, "nEv/I");
  trackTree_->Branch("nLumi", &nLumi, "nLumi/I");

  // vertex
  trackTree_->Branch("nVtx", &nVtx);
  trackTree_->Branch("xVtx", &xVtx);
  trackTree_->Branch("yVtx", &yVtx);
  trackTree_->Branch("zVtx", &zVtx);
  trackTree_->Branch("xErrVtx", &xErrVtx);
  trackTree_->Branch("yErrVtx", &yErrVtx);
  trackTree_->Branch("zErrVtx", &zErrVtx);
  trackTree_->Branch("chi2Vtx", &chi2Vtx);
  trackTree_->Branch("ndofVtx", &ndofVtx);
  trackTree_->Branch("isFakeVtx", &isFakeVtx);
  trackTree_->Branch("nTracksVtx", &nTracksVtx);
  trackTree_->Branch("ptSumVtx", &ptSumVtx);

  // Tracks
  trackTree_->Branch("nTrk", &nTrk);
  trackTree_->Branch("trkPt", &trkPt);
  trackTree_->Branch("trkPtError", &trkPtError);
  trackTree_->Branch("trkEta", &trkEta);
  trackTree_->Branch("trkPhi", &trkPhi);
  trackTree_->Branch("trkCharge", &trkCharge);
  trackTree_->Branch("trkPDGId", &trkPDGId);
  trackTree_->Branch("trkNHits", &trkNHits);
  trackTree_->Branch("trkNPixHits", &trkNPixHits);
  trackTree_->Branch("trkNLayers", &trkNLayers);
  trackTree_->Branch("trkNormChi2", &trkNormChi2);
  trackTree_->Branch("highPurity", &highPurity);

  trackTree_->Branch("pfEnergy", &pfEnergy);
  trackTree_->Branch("pfEcal", &pfEcal);
  trackTree_->Branch("pfHcal", &pfHcal);

  trackTree_->Branch("trkAssociatedVtxIndx", &trkAssociatedVtxIndx);
  trackTree_->Branch("trkAssociatedVtxQuality", &trkAssociatedVtxQuality);
  trackTree_->Branch("trkDzAssociatedVtx", &trkDzAssociatedVtx);
  trackTree_->Branch("trkDzErrAssociatedVtx", &trkDzErrAssociatedVtx);
  trackTree_->Branch("trkDxyAssociatedVtx", &trkDxyAssociatedVtx);
  trackTree_->Branch("trkDxyErrAssociatedVtx", &trkDxyErrAssociatedVtx);
  trackTree_->Branch("trkFirstVtxQuality", &trkFirstVtxQuality);
  trackTree_->Branch("trkDzFirstVtx", &trkDzFirstVtx);
  trackTree_->Branch("trkDzErrFirstVtx", &trkDzErrFirstVtx);
  trackTree_->Branch("trkDxyFirstVtx", &trkDxyFirstVtx);
  trackTree_->Branch("trkDxyErrFirstVtx", &trkDxyErrFirstVtx);
  for (const auto& d : dedxEstimatorsSrc_)
    trackTree_->Branch(d.first.c_str(), &(trkDeDx[d.first]));
}

// ------------ method called once each job just after ending the event loop  ------------
void TrackAnalyzer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackAnalyzer);
