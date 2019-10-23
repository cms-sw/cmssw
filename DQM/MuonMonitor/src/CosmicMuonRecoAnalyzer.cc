#include "DQM/MuonMonitor/interface/CosmicMuonRecoAnalyzer.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include "TMath.h"
#include "TList.h"

using namespace std;
using namespace edm;

CosmicMuonRecoAnalyzer::CosmicMuonRecoAnalyzer(const edm::ParameterSet& pSet) {
  parameters = pSet;

  // the services:
  theService = new MuonServiceProxy(parameters.getParameter<ParameterSet>("ServiceParameters"));
  theMuonCollectionLabel_ = consumes<edm::View<reco::Track> >(parameters.getParameter<edm::InputTag>("MuonCollection"));

  nTrkBin = parameters.getParameter<int>("nTrkBin");
  nTrkMax = parameters.getParameter<int>("nTrkMax");
  nTrkMin = parameters.getParameter<int>("nTrkMin");
  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");
  pBin = parameters.getParameter<int>("pBin");
  pMin = parameters.getParameter<double>("pMin");
  pMax = parameters.getParameter<double>("pMax");
  chi2Bin = parameters.getParameter<int>("chi2Bin");
  chi2Min = parameters.getParameter<double>("chi2Min");
  chi2Max = parameters.getParameter<double>("chi2Max");
  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");
  thetaBin = parameters.getParameter<int>("thetaBin");
  thetaMin = parameters.getParameter<double>("thetaMin");
  thetaMax = parameters.getParameter<double>("thetaMax");
  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");
  pResBin = parameters.getParameter<int>("pResBin");
  pResMin = parameters.getParameter<double>("pResMin");
  pResMax = parameters.getParameter<double>("pResMax");

  hitsBin = parameters.getParameter<int>("hitsBin");
  hitsMin = parameters.getParameter<int>("hitsMin");
  hitsMax = parameters.getParameter<int>("hitsMax");

  theFolder = parameters.getParameter<string>("folder");
}

CosmicMuonRecoAnalyzer::~CosmicMuonRecoAnalyzer() { delete theService; }

void CosmicMuonRecoAnalyzer::bookHistograms(DQMStore::IBooker& ibooker,
                                            edm::Run const& /*iRun*/,
                                            edm::EventSetup const& /* iSetup */) {
  ibooker.cd();
  ibooker.setCurrentFolder(theFolder);

  /////////////////////////////////////////////////////
  // monitoring of eta parameter
  /////////////////////////////////////////////////////
  std::string histname = "cosmicMuon";

  nTracksSta = ibooker.book1D(histname + "_traks", "#tracks", nTrkBin, nTrkMin, nTrkMax);

  etaStaTrack = ibooker.book1D(histname + "_eta", "#eta_{STA}", etaBin, etaMin, etaMax);

  //////////////////////////////////////////////////////
  // monitoring of theta parameter
  /////////////////////////////////////////////////////

  thetaStaTrack = ibooker.book1D(histname + "_theta", "#theta_{STA}", thetaBin, thetaMin, thetaMax);
  thetaStaTrack->setAxisTitle("rad");

  // monitoring of phi paramater

  phiStaTrack = ibooker.book1D(histname + "_phi", "#phi_{STA}", phiBin, phiMin, phiMax);
  phiStaTrack->setAxisTitle("rad");

  // monitoring of the chi2 parameter

  chi2OvDFStaTrack = ibooker.book1D(histname + "_chi2OverDf", "#chi_{2}OverDF_{STA}", chi2Bin, chi2Min, chi2Max);

  //--------------------------

  probchi2StaTrack = ibooker.book1D(histname + "_probchi", "Prob #chi_{STA}", 120, chi2Min, 1.20);

  // monitoring of the momentum

  pStaTrack = ibooker.book1D(histname + "_p", "p_{STA}", pBin, pMin, pMax);
  pStaTrack->setAxisTitle("GeV");

  qOverPStaTrack = ibooker.book1D(histname + "_qoverp", "qoverp_{STA}", pResBin, pResMin, pResMax);
  qOverPStaTrack->setAxisTitle("1/GeV");

  qOverPStaTrack_p =
      ibooker.book2D(histname + "_qoverp_p", "qoverp_p_{STA}", pBin, pMin, pMax, pResBin, pResMin, pResMax);

  // monitoring of the transverse momentum

  ptStaTrack = ibooker.book1D(histname + "_pt", "pt_{STA}", ptBin, ptMin, pMax);
  ptStaTrack->setAxisTitle("GeV");

  // monitoring of the muon charge

  qStaTrack = ibooker.book1D(histname + "_q", "q_{STA}", 5, -2.5, 2.5);

  //////////////////////////////////////////////////////////////
  // monitoring of the phi-eta

  phiVsetaStaTrack = ibooker.book2D(
      histname + "_phiVseta", "#phi vs #eta (STA)", etaBin / 2, etaMin, etaMax, phiBin / 2, phiMin, phiMax);
  phiVsetaStaTrack->setAxisTitle("eta", 1);
  phiVsetaStaTrack->setAxisTitle("phi", 2);

  // monitoring the hits
  nValidHitsStaTrack = ibooker.book1D(histname + "_nValidHits", "#valid hits (STA)", hitsBin, hitsMin, hitsMax);

  nValidHitsStaTrack_eta = ibooker.book2D(
      histname + "_nValidHits_eta", "#valid hits vs eta (STA)", etaBin, etaMin, etaMax, hitsBin, hitsMin, hitsMax);

  nValidHitsStaTrack_phi = ibooker.book2D(
      histname + "_nValidHits_phi", "#valid hits vs phi (STA)", phiBin, phiMin, phiMax, hitsBin, hitsMin, hitsMax);
}

void CosmicMuonRecoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LogTrace(metname) << "[MuonRecoAnalyzer] Analyze the mu";
  theService->update(iSetup);

  // Take the muon container
  edm::Handle<edm::View<reco::Track> > muons;
  iEvent.getByToken(theMuonCollectionLabel_, muons);

  int nTracks_ = 0;

  if (!muons.isValid())
    return;

  for (edm::View<reco::Track>::const_iterator muon = muons->begin(); muon != muons->end(); ++muon) {
    nTracks_++;

    //Needed for MVA soft muon

    // get the track using only the mu spectrometer data

    etaStaTrack->Fill(muon->eta());
    thetaStaTrack->Fill(muon->theta());
    phiStaTrack->Fill(muon->phi());
    chi2OvDFStaTrack->Fill(muon->normalizedChi2());
    probchi2StaTrack->Fill(TMath::Prob(muon->chi2(), muon->ndof()));
    pStaTrack->Fill(muon->p());
    ptStaTrack->Fill(muon->pt());
    qStaTrack->Fill(muon->charge());
    qOverPStaTrack->Fill(muon->qoverp());
    qOverPStaTrack_p->Fill(muon->p(), muon->qoverp());
    phiVsetaStaTrack->Fill(muon->eta(), muon->phi());

    nValidHitsStaTrack->Fill(muon->numberOfValidHits());
    nValidHitsStaTrack_eta->Fill(muon->eta(), muon->numberOfValidHits());
    nValidHitsStaTrack_phi->Fill(muon->phi(), muon->numberOfValidHits());
  }

  nTracksSta->Fill(nTracks_);
}
DEFINE_FWK_MODULE(CosmicMuonRecoAnalyzer);
