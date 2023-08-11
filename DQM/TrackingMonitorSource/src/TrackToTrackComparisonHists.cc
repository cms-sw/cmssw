#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/TrackingMonitorSource/interface/TrackToTrackComparisonHists.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

//
// constructors and destructor
//
TrackToTrackComparisonHists::TrackToTrackComparisonHists(const edm::ParameterSet& iConfig)
    : monitoredTrackInputTag_(iConfig.getParameter<edm::InputTag>("monitoredTrack")),
      referenceTrackInputTag_(iConfig.getParameter<edm::InputTag>("referenceTrack")),
      topDirName_(iConfig.getParameter<std::string>("topDirName")),
      dRmin_(iConfig.getParameter<double>("dRmin")),
      pTCutForPlateau_(iConfig.getParameter<double>("pTCutForPlateau")),
      dxyCutForPlateau_(iConfig.getParameter<double>("dxyCutForPlateau")),
      dzWRTPvCut_(iConfig.getParameter<double>("dzWRTPvCut")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("genericTriggerEventPSet"), consumesCollector(), *this))

{
  initialize_parameter(iConfig);

  //now do what ever initialization is needed
  monitoredTrackToken_ = consumes<reco::TrackCollection>(monitoredTrackInputTag_);
  referenceTrackToken_ = consumes<reco::TrackCollection>(referenceTrackInputTag_);
  monitoredBSToken_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("monitoredBeamSpot"));
  referenceBSToken_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("referenceBeamSpot"));
  monitoredPVToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("monitoredPrimaryVertices"));
  referencePVToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("referencePrimaryVertices"));
  lumiScalersToken_ = consumes<LumiScalersCollection>(iConfig.getParameter<edm::InputTag>("scalers"));
  onlineMetaDataDigisToken_ = consumes<OnlineLuminosityRecord>(iConfig.getParameter<edm::InputTag>("onlineMetaDataDigis"));

  referenceTracksMEs_.label = referenceTrackInputTag_.label();
  matchedReferenceTracksMEs_.label = referenceTrackInputTag_.label() + "_matched";

  monitoredTracksMEs_.label = monitoredTrackInputTag_.label();
  unMatchedMonitoredTracksMEs_.label = monitoredTrackInputTag_.label() + "_unMatched";

  matchTracksMEs_.label = "matches";
}

TrackToTrackComparisonHists::~TrackToTrackComparisonHists() {
  if (genTriggerEventFlag_)
    genTriggerEventFlag_.reset();
}

void TrackToTrackComparisonHists::beginJob(const edm::EventSetup& iSetup) {}

void TrackToTrackComparisonHists::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LogDebug("TrackToTrackComparisonHists")
      << " requireValidHLTPaths_ " << requireValidHLTPaths_ << " hltPathsAreValid_  " << hltPathsAreValid_ << "\n";
  // if valid HLT paths are required,
  // analyze event only if paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  LogDebug("TrackToTrackComparisonHists") << " genTriggerEventFlag_->on() " << genTriggerEventFlag_->on()
                                          << "  accept:  " << genTriggerEventFlag_->accept(iEvent, iSetup) << "\n";
  // Filter out events if Trigger Filtering is requested
  if (genTriggerEventFlag_->on() && !genTriggerEventFlag_->accept(iEvent, iSetup)) {
    return;
  }


  //
  //  Get Lumi/LS Info
  //

  unsigned int ls = iEvent.id().luminosityBlock();

  double onlinelumi = -1.f;
  double PU = -1.f;

  auto const lumiScalersHandle = iEvent.getHandle(lumiScalersToken_);
  auto const onlineMetaDataDigisHandle = iEvent.getHandle(onlineMetaDataDigisToken_);

  if (onlineMetaDataDigisHandle.isValid()) {
    onlinelumi = onlineMetaDataDigisHandle->instLumi();
    PU = onlineMetaDataDigisHandle->avgPileUp();
  } else if ( lumiScalersHandle.isValid() and not lumiScalersHandle->empty() ){
    edm::LogError("TrackToTrackComparisonHists") << "onlineMetaDataDigisHandle not found, trying SCAL";
    auto const scalit = lumiScalersHandle->begin();
    onlinelumi = scalit->instantLumi();
    PU = scalit->pileup();
  } else {
    edm::LogError("TrackToTrackComparisonHists") << "lumiScalersHandle not found or empty, skipping event";
    return;
  }

  //
  //  Get Reference Track Info
  //
  edm::Handle<reco::TrackCollection> referenceTracksHandle;
  iEvent.getByToken(referenceTrackToken_, referenceTracksHandle);
  if (!referenceTracksHandle.isValid()) {
    edm::LogError("TrackToTrackComparisonHists") << "referenceTracksHandle not found, skipping event";
    return;
  }
  reco::TrackCollection referenceTracks = *referenceTracksHandle;

  edm::Handle<reco::BeamSpot> referenceBSHandle;
  iEvent.getByToken(referenceBSToken_, referenceBSHandle);
  if (!referenceBSHandle.isValid()) {
    edm::LogError("TrackToTrackComparisonHists") << "referenceBSHandle not found, skipping event";
    return;
  }
  reco::BeamSpot referenceBS = *referenceBSHandle;

  edm::Handle<reco::VertexCollection> referencePVHandle;
  iEvent.getByToken(referencePVToken_, referencePVHandle);
  if (!referencePVHandle.isValid()) {
    edm::LogError("TrackToTrackComparisonHists") << "referencePVHandle not found, skipping event";
    return;
  }
  if (referencePVHandle->empty()) {
    edm::LogInfo("TrackToTrackComparisonHists") << "referencePVHandle->size is 0 ";
    return;
  }
  reco::Vertex referencePV = referencePVHandle->at(0);

  //
  //  Get Monitored Track Info
  //
  edm::Handle<reco::TrackCollection> monitoredTracksHandle;
  iEvent.getByToken(monitoredTrackToken_, monitoredTracksHandle);
  if (!monitoredTracksHandle.isValid()) {
    edm::LogError("TrackToTrackComparisonHists") << "monitoredTracksHandle not found, skipping event";
    return;
  }
  reco::TrackCollection monitoredTracks = *monitoredTracksHandle;

  edm::Handle<reco::BeamSpot> monitoredBSHandle;
  iEvent.getByToken(monitoredBSToken_, monitoredBSHandle);
  if (!monitoredTracksHandle.isValid()) {
    edm::LogError("TrackToTrackComparisonHists") << "monitoredBSHandle not found, skipping event";
    return;
  }
  reco::BeamSpot monitoredBS = *monitoredBSHandle;

  edm::Handle<reco::VertexCollection> monitoredPVHandle;
  iEvent.getByToken(monitoredPVToken_, monitoredPVHandle);
  if (!monitoredPVHandle.isValid()) {
    edm::LogError("TrackToTrackComparisonHists") << "monitoredPVHandle not found, skipping event";
    return;
  }
  if (monitoredPVHandle->empty()) {
    edm::LogInfo("TrackToTrackComparisonHists") << "monitoredPVHandle->size is 0 ";
    return;
  }
  reco::Vertex monitoredPV = monitoredPVHandle->at(0);

  edm::LogInfo("TrackToTrackComparisonHists")
      << "analyzing " << monitoredTrackInputTag_.process() << ":" << monitoredTrackInputTag_.label() << ":"
      << monitoredTrackInputTag_.instance() << " w.r.t. " << referenceTrackInputTag_.process() << ":"
      << referenceTrackInputTag_.label() << ":" << referenceTrackInputTag_.instance() << " \n";

  //
  // Build the dR maps
  //
  idx2idxByDoubleColl monitored2referenceColl;
  fillMap(monitoredTracks, referenceTracks, monitored2referenceColl, dRmin_);

  idx2idxByDoubleColl reference2monitoredColl;
  fillMap(referenceTracks, monitoredTracks, reference2monitoredColl, dRmin_);

  unsigned int nReferenceTracks(0);           // Counts the number of refernce tracks
  unsigned int nMatchedReferenceTracks(0);    // Counts the number of matched refernce tracks
  unsigned int nMonitoredTracks(0);           // Counts the number of monitored tracks
  unsigned int nUnmatchedMonitoredTracks(0);  // Counts the number of unmatched monitored tracks

  //
  // loop over reference tracks
  //
  LogDebug("TrackToTrackComparisonHists") << "\n# of tracks (reference): " << referenceTracks.size() << "\n";
  for (idx2idxByDoubleColl::const_iterator pItr = reference2monitoredColl.begin(), eItr = reference2monitoredColl.end();
       pItr != eItr;
       ++pItr) {
    nReferenceTracks++;
    int trackIdx = pItr->first;
    reco::Track track = referenceTracks.at(trackIdx);

    float dzWRTpv = track.dz(referencePV.position());
    if (fabs(dzWRTpv) > dzWRTPvCut_)
      continue;

    fill_generic_tracks_histos(*&referenceTracksMEs_, &track, &referenceBS, &referencePV, ls, onlinelumi, PU);

    std::map<double, int> trackDRmap = pItr->second;
    if (trackDRmap.empty()) {
      (matchedReferenceTracksMEs_.h_dRmin)->Fill(-1.);
      (matchedReferenceTracksMEs_.h_dRmin_l)->Fill(-1.);
      continue;
    }

    double dRmin = trackDRmap.begin()->first;
    (referenceTracksMEs_.h_dRmin)->Fill(dRmin);
    (referenceTracksMEs_.h_dRmin_l)->Fill(dRmin);

    bool matched = false;
    if (dRmin < dRmin_)
      matched = true;

    if (matched) {
      nMatchedReferenceTracks++;
      fill_generic_tracks_histos(*&matchedReferenceTracksMEs_, &track, &referenceBS, &referencePV, ls, onlinelumi, PU);
      (matchedReferenceTracksMEs_.h_dRmin)->Fill(dRmin);
      (matchedReferenceTracksMEs_.h_dRmin_l)->Fill(dRmin);

      int matchedTrackIndex = trackDRmap[dRmin];
      reco::Track matchedTrack = monitoredTracks.at(matchedTrackIndex);
      fill_matching_tracks_histos(*&matchTracksMEs_, &track, &matchedTrack, &referenceBS, &referencePV);
    }

  }  // Over reference tracks

  //
  // loop over monitoed tracks
  //
  LogDebug("TrackToTrackComparisonHists") << "\n# of tracks (monitored): " << monitoredTracks.size() << "\n";
  for (idx2idxByDoubleColl::const_iterator pItr = monitored2referenceColl.begin(), eItr = monitored2referenceColl.end();
       pItr != eItr;
       ++pItr) {
    nMonitoredTracks++;
    int trackIdx = pItr->first;
    reco::Track track = monitoredTracks.at(trackIdx);

    float dzWRTpv = track.dz(monitoredPV.position());
    if (fabs(dzWRTpv) > dzWRTPvCut_)
      continue;

    fill_generic_tracks_histos(*&monitoredTracksMEs_, &track, &monitoredBS, &monitoredPV, ls, onlinelumi, PU);

    std::map<double, int> trackDRmap = pItr->second;
    if (trackDRmap.empty()) {
      (unMatchedMonitoredTracksMEs_.h_dRmin)->Fill(-1.);
      (unMatchedMonitoredTracksMEs_.h_dRmin_l)->Fill(-1.);
      continue;
    }

    double dRmin = trackDRmap.begin()->first;
    (monitoredTracksMEs_.h_dRmin)->Fill(dRmin);
    (monitoredTracksMEs_.h_dRmin_l)->Fill(dRmin);

    bool matched = false;
    if (dRmin < dRmin_)
      matched = true;

    if (!matched) {
      nUnmatchedMonitoredTracks++;
      fill_generic_tracks_histos(*&unMatchedMonitoredTracksMEs_, &track, &monitoredBS, &monitoredPV, ls, onlinelumi, PU);
      (unMatchedMonitoredTracksMEs_.h_dRmin)->Fill(dRmin);
      (unMatchedMonitoredTracksMEs_.h_dRmin_l)->Fill(dRmin);
    }

  }  // over monitoed tracks

  edm::LogInfo("TrackToTrackComparisonHists")
      << "Total reference tracks: " << nReferenceTracks << "\n"
      << "Total matched reference tracks: " << nMatchedReferenceTracks << "\n"
      << "Total monitored tracks: " << nMonitoredTracks << "\n"
      << "Total unMatched monitored tracks: " << nUnmatchedMonitoredTracks << "\n";

}

void TrackToTrackComparisonHists::bookHistograms(DQMStore::IBooker& ibooker,
                                                 edm::Run const& iRun,
                                                 edm::EventSetup const& iSetup) {
  if (genTriggerEventFlag_ && genTriggerEventFlag_->on())
    genTriggerEventFlag_->initRun(iRun, iSetup);

  // check if every HLT path specified has a valid match in the HLT Menu
  hltPathsAreValid_ =
      (genTriggerEventFlag_ && genTriggerEventFlag_->on() && genTriggerEventFlag_->allHLTPathsAreValid());

  // if valid HLT paths are required,
  // create DQM outputs only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  std::string dir = topDirName_;

  bookHistos(ibooker, referenceTracksMEs_, "ref", dir);
  bookHistos(ibooker, matchedReferenceTracksMEs_, "ref_matched", dir);

  bookHistos(ibooker, monitoredTracksMEs_, "mon", dir);
  bookHistos(ibooker, unMatchedMonitoredTracksMEs_, "mon_unMatched", dir);

  book_matching_tracks_histos(ibooker, matchTracksMEs_, "matches", dir);

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TrackToTrackComparisonHists::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("requireValidHLTPaths", true);

  desc.add<edm::InputTag>("monitoredTrack", edm::InputTag("hltMergedTracks"));
  desc.add<edm::InputTag>("monitoredBeamSpot", edm::InputTag("hltOnlineBeamSpot"));
  desc.add<edm::InputTag>("monitoredPrimaryVertices", edm::InputTag("hltVerticesPFSelector"));

  desc.add<edm::InputTag>("referenceTrack", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("referenceBeamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("referencePrimaryVertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("scalers", edm::InputTag("scalersRawToDigi"));
  desc.add<edm::InputTag>("onlineMetaDataDigis", edm::InputTag("onlineMetaDataDigis"));

  desc.add<std::string>("topDirName", "HLT/Tracking/ValidationWRTOffline");
  desc.add<double>("dRmin", 0.002);

  desc.add<double>("pTCutForPlateau", 0.9);
  desc.add<double>("dxyCutForPlateau", 2.5);
  desc.add<double>("dzWRTPvCut", 1e6);

  edm::ParameterSetDescription genericTriggerEventPSet;
  GenericTriggerEventFlag::fillPSetDescription(genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("genericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  fillHistoPSetDescription(histoPSet);
  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);


  descriptions.add("trackToTrackComparisonHists", desc);
}

void TrackToTrackComparisonHists::fillMap(reco::TrackCollection tracks1,
                                          reco::TrackCollection tracks2,
                                          idx2idxByDoubleColl& map,
                                          float dRMin) {
  //
  // loop on tracks1
  //
  int i = 0;
  for (const auto& track1 : tracks1) {
    std::map<double, int> tmp;
    int j = 0;
    float smallest_dR = 1e9;
    int smallest_dR_j = -1;

    //
    // loop on tracks2
    //
    for (const auto& track2 : tracks2) {
      double dR = reco::deltaR(track1.eta(), track1.phi(), track2.eta(), track2.phi());

      if (dR < smallest_dR) {
        smallest_dR = dR;
        smallest_dR_j = j;
      }

      if (dR < dRMin) {
        tmp[dR] = j;
      }

      j++;
    }

    //
    // If there are no tracks that pass the dR store the smallest (for debugging/validating matching)
    //
    if (tmp.empty())
      tmp[smallest_dR] = smallest_dR_j;

    map.push_back(std::make_pair(i, tmp));
    i++;
  }
}

void TrackToTrackComparisonHists::bookHistos(DQMStore::IBooker& ibooker,
                                             generalME& mes,
                                             TString label,
                                             std::string& dir) {
  book_generic_tracks_histos(ibooker, mes, label, dir);
}

void TrackToTrackComparisonHists::book_generic_tracks_histos(DQMStore::IBooker& ibooker,
                                                             generalME& mes,
                                                             TString label,
                                                             std::string& dir) {
  ibooker.cd();
  ibooker.setCurrentFolder(dir);
  (mes.h_pt) = ibooker.book1D(label + "_pt", "track p_{T}", Pt_nbin, Pt_rangeMin, Pt_rangeMax);
  (mes.h_eta) = ibooker.book1D(label + "_eta", "track pseudorapidity", Eta_nbin, Eta_rangeMin, Eta_rangeMax);
  (mes.h_phi) = ibooker.book1D(label + "_phi", "track #phi", Phi_nbin, Phi_rangeMin, Phi_rangeMax);
  (mes.h_dxy) =
      ibooker.book1D(label + "_dxy", "track transverse dca to beam spot", Dxy_nbin, Dxy_rangeMin, Dxy_rangeMax);
  (mes.h_dz) = ibooker.book1D(label + "_dz", "track longitudinal dca to beam spot", Dz_nbin, Dz_rangeMin, Dz_rangeMax);
  (mes.h_dxyWRTpv) = ibooker.book1D(
      label + "_dxyWRTpv", "track transverse dca to primary vertex", Dxy_nbin, Dxy_rangeMin, Dxy_rangeMax);
  (mes.h_dzWRTpv) = ibooker.book1D(
      label + "_dzWRTpv", "track longitudinal dca to primary vertex", Dz_nbin, 0.1 * Dz_rangeMin, 0.1 * Dz_rangeMax);
  (mes.h_charge) = ibooker.book1D(label + "_charge", "track charge", 5, -2, 2);
  (mes.h_hits) = ibooker.book1D(label + "_hits", "track number of hits", 35, -0.5, 34.5);
  (mes.h_dRmin) = ibooker.book1D(label + "_dRmin", "track min dR", 100, 0., 0.01);
  (mes.h_dRmin_l) = ibooker.book1D(label + "_dRmin_l", "track min dR", 100, 0., 0.4);

  (mes.h_pt_vs_eta) = ibooker.book2D(label + "_ptVSeta",
                                     "track p_{T} vs #eta",
                                     Eta_nbin,
                                     Eta_rangeMin,
                                     Eta_rangeMax,
                                     Pt_nbin,
                                     Pt_rangeMin,
                                     Pt_rangeMax);
                        
  // counts of tracks vs lumi
  // for this moment, xmin,xmax and binning are hardcoded, maybe in future in a config file!
  // have to add (declare) this in the .h file as well
  (mes.h_onlinelumi) = ibooker.book1D(label + "_onlinelumi", "number of tracks vs onlinelumi", onlinelumi_nbin, onlinelumi_rangeMin, onlinelumi_rangeMax);
  (mes.h_ls) = ibooker.book1D(label + "_ls", "number of tracks vs ls", ls_nbin, ls_rangeMin, ls_rangeMax);
  (mes.h_PU) = ibooker.book1D(label + "_PU", "number of tracks vs PU", PU_nbin, PU_rangeMin, PU_rangeMax);

}

void TrackToTrackComparisonHists::book_matching_tracks_histos(DQMStore::IBooker& ibooker,
                                                              matchingME& mes,
                                                              TString label,
                                                              std::string& dir) {
  ibooker.cd();
  ibooker.setCurrentFolder(dir);

  (mes.h_hits_vs_hits) = ibooker.book2D(
      label + "_hits_vs_hits", "monitored track # hits vs reference track # hits", 35, -0.5, 34.5, 35, -0.5, 34.5);
  (mes.h_pt_vs_pt) = ibooker.book2D(label + "_pt_vs_pt",
                                    "monitored track p_{T} vs reference track p_{T}",
                                    Pt_nbin,
                                    Pt_rangeMin,
                                    Pt_rangeMax,
                                    Pt_nbin,
                                    Pt_rangeMin,
                                    Pt_rangeMax);
  (mes.h_eta_vs_eta) = ibooker.book2D(label + "_eta_vs_eta",
                                      "monitored track #eta vs reference track #eta",
                                      Eta_nbin,
                                      Eta_rangeMin,
                                      Eta_rangeMax,
                                      Eta_nbin,
                                      Eta_rangeMin,
                                      Eta_rangeMax);
  (mes.h_phi_vs_phi) = ibooker.book2D(label + "_phi_vs_phi",
                                      "monitored track #phi vs reference track #phi",
                                      Phi_nbin,
                                      Phi_rangeMin,
                                      Phi_rangeMax,
                                      Phi_nbin,
                                      Phi_rangeMin,
                                      Phi_rangeMax);

  (mes.h_dPt) = ibooker.book1D(label + "_dPt", "#Delta track #P_T", ptRes_nbin, ptRes_rangeMin, ptRes_rangeMax);
  (mes.h_dEta) = ibooker.book1D(label + "_dEta", "#Delta track #eta", etaRes_nbin, etaRes_rangeMin, etaRes_rangeMax);
  (mes.h_dPhi) = ibooker.book1D(label + "_dPhi", "#Delta track #phi", phiRes_nbin, phiRes_rangeMin, phiRes_rangeMax);
  (mes.h_dDxy) = ibooker.book1D(
      label + "_dDxy", "#Delta track transverse dca to beam spot", dxyRes_nbin, dxyRes_rangeMin, dxyRes_rangeMax);
  (mes.h_dDz) = ibooker.book1D(
      label + "_dDz", "#Delta track longitudinal dca to beam spot", dzRes_nbin, dzRes_rangeMin, dzRes_rangeMax);
  (mes.h_dDxyWRTpv) = ibooker.book1D(label + "_dDxyWRTpv",
                                     "#Delta track transverse dca to primary vertex ",
                                     dxyRes_nbin,
                                     dxyRes_rangeMin,
                                     dxyRes_rangeMax);
  (mes.h_dDzWRTpv) = ibooker.book1D(label + "_dDzWRTpv",
                                    "#Delta track longitudinal dca to primary vertex",
                                    dzRes_nbin,
                                    dzRes_rangeMin,
                                    dzRes_rangeMax);
  (mes.h_dCharge) = ibooker.book1D(label + "_dCharge", "#Delta track charge", 5, -2.5, 2.5);
  (mes.h_dHits) = ibooker.book1D(label + "_dHits", "#Delta track number of hits", 39, -19.5, 19.5);
}

void TrackToTrackComparisonHists::fill_generic_tracks_histos(
    generalME& mes, reco::Track* trk, reco::BeamSpot* bs, reco::Vertex* pv, unsigned int ls, double onlinelumi, double PU, bool requirePlateau) {
  float pt = trk->pt();
  float eta = trk->eta();
  float phi = trk->phi();
  float dxy = trk->dxy(bs->position());
  float dz = trk->dz(bs->position());
  float dxyWRTpv = trk->dxy(pv->position());
  float dzWRTpv = trk->dz(pv->position());
  float charge = trk->charge();
  float nhits = trk->hitPattern().numberOfValidHits();

  bool dxyOnPlateau = (fabs(dxyWRTpv) < dxyCutForPlateau_);
  bool pTOnPlateau = (pt > pTCutForPlateau_);

  if (dxyOnPlateau || !requirePlateau) {
    (mes.h_pt)->Fill(pt);
  }

  if ((pTOnPlateau && dxyOnPlateau) || !requirePlateau) {
    (mes.h_eta)->Fill(eta);
    (mes.h_phi)->Fill(phi);
    (mes.h_dz)->Fill(dz);
    (mes.h_dzWRTpv)->Fill(dzWRTpv);
    (mes.h_charge)->Fill(charge);
    (mes.h_hits)->Fill(nhits);
    (mes.h_onlinelumi)->Fill(onlinelumi);
    (mes.h_ls)->Fill(ls);
    (mes.h_PU)->Fill(PU);

  }

  if (pTOnPlateau || !requirePlateau) {
    (mes.h_dxy)->Fill(dxy);
    (mes.h_dxyWRTpv)->Fill(dxyWRTpv);
  }

  (mes.h_pt_vs_eta)->Fill(eta, pt);
}

void TrackToTrackComparisonHists::fill_matching_tracks_histos(
    matchingME& mes, reco::Track* mon, reco::Track* ref, reco::BeamSpot* bs, reco::Vertex* pv) {
  float mon_pt = mon->pt();
  float mon_eta = mon->eta();
  float mon_phi = mon->phi();
  float mon_dxy = mon->dxy(bs->position());
  float mon_dz = mon->dz(bs->position());
  float mon_dxyWRTpv = mon->dxy(pv->position());
  float mon_dzWRTpv = mon->dz(pv->position());
  float mon_charge = mon->charge();
  float mon_nhits = mon->hitPattern().numberOfValidHits();

  float ref_pt = ref->pt();
  float ref_eta = ref->eta();
  float ref_phi = ref->phi();
  float ref_dxy = ref->dxy(bs->position());
  float ref_dz = ref->dz(bs->position());
  float ref_dxyWRTpv = ref->dxy(pv->position());
  float ref_dzWRTpv = ref->dz(pv->position());
  float ref_charge = ref->charge();
  float ref_nhits = ref->hitPattern().numberOfValidHits();

  (mes.h_hits_vs_hits)->Fill(ref_nhits, mon_nhits);
  (mes.h_pt_vs_pt)->Fill(ref_pt, mon_pt);
  (mes.h_eta_vs_eta)->Fill(ref_eta, mon_eta);
  (mes.h_phi_vs_phi)->Fill(ref_phi, mon_phi);

  (mes.h_dPt)->Fill(ref_pt - mon_pt);
  (mes.h_dEta)->Fill(ref_eta - mon_eta);
  (mes.h_dPhi)->Fill(ref_phi - mon_phi);
  (mes.h_dDxy)->Fill(ref_dxy - mon_dxy);
  (mes.h_dDz)->Fill(ref_dz - mon_dz);
  (mes.h_dDxyWRTpv)->Fill(ref_dxyWRTpv - mon_dxyWRTpv);
  (mes.h_dDzWRTpv)->Fill(ref_dzWRTpv - mon_dzWRTpv);
  (mes.h_dCharge)->Fill(ref_charge - mon_charge);
  (mes.h_dHits)->Fill(ref_nhits - mon_nhits);
}

void TrackToTrackComparisonHists::initialize_parameter(const edm::ParameterSet& iConfig) {
  const edm::ParameterSet& pset = iConfig.getParameter<edm::ParameterSet>("histoPSet");

  Eta_rangeMin = pset.getParameter<double>("Eta_rangeMin");
  Eta_rangeMax = pset.getParameter<double>("Eta_rangeMax");
  Eta_nbin = pset.getParameter<unsigned int>("Eta_nbin");

  Pt_rangeMin = pset.getParameter<double>("Pt_rangeMin");
  Pt_rangeMax = pset.getParameter<double>("Pt_rangeMax");
  Pt_nbin = pset.getParameter<unsigned int>("Pt_nbin");

  Phi_rangeMin = pset.getParameter<double>("Phi_rangeMin");
  Phi_rangeMax = pset.getParameter<double>("Phi_rangeMax");
  Phi_nbin = pset.getParameter<unsigned int>("Phi_nbin");

  Dxy_rangeMin = pset.getParameter<double>("Dxy_rangeMin");
  Dxy_rangeMax = pset.getParameter<double>("Dxy_rangeMax");
  Dxy_nbin = pset.getParameter<unsigned int>("Dxy_nbin");

  Dz_rangeMin = pset.getParameter<double>("Dz_rangeMin");
  Dz_rangeMax = pset.getParameter<double>("Dz_rangeMax");
  Dz_nbin = pset.getParameter<unsigned int>("Dz_nbin");

  ptRes_rangeMin = pset.getParameter<double>("ptRes_rangeMin");
  ptRes_rangeMax = pset.getParameter<double>("ptRes_rangeMax");
  ptRes_nbin = pset.getParameter<unsigned int>("ptRes_nbin");

  phiRes_rangeMin = pset.getParameter<double>("phiRes_rangeMin");
  phiRes_rangeMax = pset.getParameter<double>("phiRes_rangeMax");
  phiRes_nbin = pset.getParameter<unsigned int>("phiRes_nbin");

  etaRes_rangeMin = pset.getParameter<double>("etaRes_rangeMin");
  etaRes_rangeMax = pset.getParameter<double>("etaRes_rangeMax");
  etaRes_nbin = pset.getParameter<unsigned int>("etaRes_nbin");

  dxyRes_rangeMin = pset.getParameter<double>("dxyRes_rangeMin");
  dxyRes_rangeMax = pset.getParameter<double>("dxyRes_rangeMax");
  dxyRes_nbin = pset.getParameter<unsigned int>("dxyRes_nbin");

  dzRes_rangeMin = pset.getParameter<double>("dzRes_rangeMin");
  dzRes_rangeMax = pset.getParameter<double>("dzRes_rangeMax");
  dzRes_nbin = pset.getParameter<unsigned int>("dzRes_nbin");

  ls_rangeMin = pset.getParameter<unsigned int>("ls_rangeMin");
  ls_rangeMax = pset.getParameter<unsigned int>("ls_rangeMax");
  ls_nbin = pset.getParameter<unsigned int>("ls_nbin");

  onlinelumi_rangeMin = pset.getParameter<double>("onlinelumi_rangeMin");
  onlinelumi_rangeMax = pset.getParameter<double>("onlinelumi_rangeMax");
  onlinelumi_nbin = pset.getParameter<unsigned int>("onlinelumi_nbin");

  PU_rangeMin = pset.getParameter<double>("PU_rangeMin");
  PU_rangeMax = pset.getParameter<double>("PU_rangeMax");
  PU_nbin = pset.getParameter<unsigned int>("PU_nbin");
}

void TrackToTrackComparisonHists::fillHistoPSetDescription(edm::ParameterSetDescription& pset) {
  pset.add<double>("Eta_rangeMin", -2.5);
  pset.add<double>("Eta_rangeMax", 2.5);
  pset.add<unsigned int>("Eta_nbin", 50);

  pset.add<double>("Pt_rangeMin", 0.1);
  pset.add<double>("Pt_rangeMax", 100.0);
  pset.add<unsigned int>("Pt_nbin", 1000);

  pset.add<double>("Phi_rangeMin", -3.1416);
  pset.add<double>("Phi_rangeMax", 3.1416);
  pset.add<unsigned int>("Phi_nbin", 36);

  pset.add<double>("Dxy_rangeMin", -1.0);
  pset.add<double>("Dxy_rangeMax", 1.0);
  pset.add<unsigned int>("Dxy_nbin", 300);

  pset.add<double>("Dz_rangeMin", -30.0);
  pset.add<double>("Dz_rangeMax", 30.0);
  pset.add<unsigned int>("Dz_nbin", 60);

  pset.add<double>("ptRes_rangeMin", -0.1);
  pset.add<double>("ptRes_rangeMax", 0.1);
  pset.add<unsigned int>("ptRes_nbin", 100);

  pset.add<double>("phiRes_rangeMin", -0.01);
  pset.add<double>("phiRes_rangeMax", 0.01);
  pset.add<unsigned int>("phiRes_nbin", 300);

  pset.add<double>("etaRes_rangeMin", -0.01);
  pset.add<double>("etaRes_rangeMax", 0.01);
  pset.add<unsigned int>("etaRes_nbin", 300);

  pset.add<double>("dxyRes_rangeMin", -0.05);
  pset.add<double>("dxyRes_rangeMax", 0.05);
  pset.add<unsigned int>("dxyRes_nbin", 500);

  pset.add<double>("dzRes_rangeMin", -0.05);
  pset.add<double>("dzRes_rangeMax", 0.05);
  pset.add<unsigned int>("dzRes_nbin", 150);

  pset.add<unsigned int>("ls_rangeMin",0);
  pset.add<unsigned int>("ls_rangeMax",3000);
  pset.add<unsigned int>("ls_nbin",300);

  pset.add<double>("onlinelumi_rangeMin",0.0);
  pset.add<double>("onlinelumi_rangeMax",20000.0);
  pset.add<unsigned int>("onlinelumi_nbin",200);

  pset.add<double>("PU_rangeMin",0.0);
  pset.add<double>("PU_rangeMax",120.0);
  pset.add<unsigned int>("PU_nbin",120);
}
