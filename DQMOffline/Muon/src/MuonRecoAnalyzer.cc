
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/28 15:21:03 $
 *  $Revision: 1.6 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/src/MuonRecoAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;



MuonRecoAnalyzer::MuonRecoAnalyzer(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService) {

 cout<<"[MuonRecoAnalyzer] Constructor called!"<<endl;
  parameters = pSet;

}


MuonRecoAnalyzer::~MuonRecoAnalyzer() { }


void MuonRecoAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  metname = "muRecoAnalyzer";

  LogTrace(metname)<<"[MuonRecoAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("Muons/MuonRecoAnalyzer");

  muReco = dbe->book1D("muReco", "muReco", 3, 0, 3);
  muReco->setBinLabel(1,"global",1);
  muReco->setBinLabel(2,"tracker",1);
  muReco->setBinLabel(3,"sta",1);

  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");
  std::string histname = "etaGlbTrack_";
  etaGlbTrack.push_back(dbe->book1D(histname+"combined", histname+"combined", etaBin, etaMin, etaMax));
  etaGlbTrack.push_back(dbe->book1D(histname+"tracker", histname+"tracker", etaBin, etaMin, etaMax));
  etaGlbTrack.push_back(dbe->book1D(histname+"sta", histname+"sta", etaBin, etaMin, etaMax));
  etaResolution.push_back(dbe->book1D(histname+"trackerResolution", histname+"trackerResolution", etaBin, etaMin, etaMax));
  etaResolution.push_back(dbe->book1D(histname+"staResolution", histname+"staResolution", etaBin, etaMin, etaMax));
  etaTrack = dbe->book1D("etaTrack", "etaTrack", etaBin, etaMin, etaMax);
  etaStaTrack = dbe->book1D("etaStaTrack", "etaStaTrack", etaBin, etaMin, etaMax);

  thetaBin = parameters.getParameter<int>("thetaBin");
  thetaMin = parameters.getParameter<double>("thetaMin");
  thetaMax = parameters.getParameter<double>("thetaMax");
  histname = "thetaGlbTrack_";
  thetaGlbTrack.push_back(dbe->book1D(histname+"combined", histname+"combined", thetaBin, thetaMin, thetaMax));
  thetaGlbTrack.push_back(dbe->book1D(histname+"tracker", histname+"tracker", thetaBin, thetaMin, thetaMax));
  thetaGlbTrack.push_back(dbe->book1D(histname+"sta", histname+"sta", thetaBin, thetaMin, thetaMax));
  thetaResolution.push_back(dbe->book1D(histname+"trackerResolution", histname+"trackerResolution", thetaBin, thetaMin, thetaMax));
  thetaResolution.push_back(dbe->book1D(histname+"staResolution", histname+"staResolution", thetaBin, thetaMin, thetaMax));
  thetaTrack = dbe->book1D("thetaTrack", "thetaTrack", thetaBin, thetaMin, thetaMax);
  thetaStaTrack = dbe->book1D("thetaStaTrack", "thetaStaTrack", thetaBin, thetaMin, thetaMax);

  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");
  histname = "phiGlbTrack_";
  phiGlbTrack.push_back(dbe->book1D(histname+"combined", histname+"combined", phiBin, phiMin, phiMax));
  phiGlbTrack.push_back(dbe->book1D(histname+"tracker", histname+"tracker", phiBin, phiMin, phiMax));
  phiGlbTrack.push_back(dbe->book1D(histname+"sta", histname+"sta", phiBin, phiMin, phiMax));
  phiResolution.push_back(dbe->book1D(histname+"trackerResolution", histname+"trackerResolution", phiBin, phiMin, phiMax));
  phiResolution.push_back(dbe->book1D(histname+"staResolution", histname+"staResolution", phiBin, phiMin, phiMax));
  phiTrack = dbe->book1D("phiTrack", "phiTrack", phiBin, phiMin, phiMax);
  phiStaTrack = dbe->book1D("phiStaTrack", "phiStaTrack", phiBin, phiMin, phiMax);

  qOverpBin = parameters.getParameter<int>("qOverpBin");
  qOverpMin = parameters.getParameter<double>("qOverpMin");
  qOverpMax = parameters.getParameter<double>("qOverpMax");
  histname = "qOverpGlbTrack_";
  qOverpGlbTrack.push_back(dbe->book1D(histname+"combined", histname+"combined", qOverpBin, qOverpMin, qOverpMax));
  qOverpGlbTrack.push_back(dbe->book1D(histname+"tracker", histname+"tracker", qOverpBin, qOverpMin, qOverpMax));
  qOverpGlbTrack.push_back(dbe->book1D(histname+"sta", histname+"sta", qOverpBin, qOverpMin, qOverpMax));
  qOverpResolution.push_back(dbe->book1D(histname+"trackerResolution", histname+"trackerResolution", qOverpBin, qOverpMin, qOverpMax));
  qOverpResolution.push_back(dbe->book1D(histname+"staResolution", histname+"staResolution", qOverpBin, qOverpMin, qOverpMax));
  qOverpTrack = dbe->book1D("qOverpTrack", "qOverpTrack", qOverpBin, qOverpMin, qOverpMax);
  qOverpStaTrack = dbe->book1D("qOverpStaTrack", "qOverpStaTrack", qOverpBin, qOverpMin, qOverpMax);

  qOverptBin = parameters.getParameter<int>("qOverptBin");
  qOverptMin = parameters.getParameter<double>("qOverptMin");
  qOverptMax = parameters.getParameter<double>("qOverptMax");
  histname = "qOverptGlbTrack_";
  qOverptGlbTrack.push_back(dbe->book1D(histname+"combined", histname+"combined", qOverptBin, qOverptMin, qOverptMax));
  qOverptGlbTrack.push_back(dbe->book1D(histname+"tracker", histname+"tracker", qOverptBin, qOverptMin, qOverptMax));
  qOverptGlbTrack.push_back(dbe->book1D(histname+"sta", histname+"sta", qOverptBin, qOverptMin, qOverptMax));
  qOverptResolution.push_back(dbe->book1D(histname+"trackerResolution", histname+"trackerResoution", qOverptBin, qOverptMin, qOverptMax));
  qOverptResolution.push_back(dbe->book1D(histname+"staResolution", histname+"staResoution", qOverptBin, qOverptMin, qOverptMax));
  qOverptTrack = dbe->book1D("qOverptTrack", "qOverptTrack", qOverptBin, qOverptMin, qOverptMax);
  qOverptStaTrack = dbe->book1D("qOverptStaTrack", "qOverptStaTrack", qOverptBin, qOverptMin, qOverptMax);

}


void MuonRecoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Muon& recoMu) {

  LogTrace(metname)<<"[MuonRecoAnalyzer] Analyze the mu";

  if(recoMu.isGlobalMuon()) {

    LogTrace(metname)<<"[MuonRecoAnalyzer] The mu is global - filling the histos";
    muReco->Fill(1);

    // get the track combinig the information from both the Tracker and the Spectrometer
    reco::TrackRef recoCombinedGlbTrack = recoMu.combinedMuon();
    // get the track using only the tracker data
    reco::TrackRef recoGlbTrack = recoMu.track();
    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaGlbTrack = recoMu.standAloneMuon();
  
    etaGlbTrack[0]->Fill(recoCombinedGlbTrack->eta());
    etaGlbTrack[1]->Fill(recoGlbTrack->eta());
    etaGlbTrack[2]->Fill(recoStaGlbTrack->eta());
    etaResolution[0]->Fill(recoGlbTrack->eta()-recoCombinedGlbTrack->eta());
    etaResolution[1]->Fill(recoStaGlbTrack->eta()-recoCombinedGlbTrack->eta());

    thetaGlbTrack[0]->Fill(recoCombinedGlbTrack->theta());
    thetaGlbTrack[1]->Fill(recoGlbTrack->theta());
    thetaGlbTrack[2]->Fill(recoStaGlbTrack->theta());
    thetaResolution[0]->Fill(recoGlbTrack->theta()-recoCombinedGlbTrack->theta());
    thetaResolution[1]->Fill(recoStaGlbTrack->theta()-recoCombinedGlbTrack->theta());
    
    phiGlbTrack[0]->Fill(recoCombinedGlbTrack->phi());
    phiGlbTrack[1]->Fill(recoGlbTrack->phi());
    phiGlbTrack[2]->Fill(recoStaGlbTrack->phi());
    phiResolution[0]->Fill(recoGlbTrack->phi()-recoCombinedGlbTrack->phi());
    phiResolution[1]->Fill(recoStaGlbTrack->phi()-recoCombinedGlbTrack->phi());
    
    qOverpGlbTrack[0]->Fill(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->p());
    qOverpGlbTrack[1]->Fill(recoGlbTrack->charge()/recoGlbTrack->p());
    qOverpGlbTrack[2]->Fill(recoStaGlbTrack->charge()/recoStaGlbTrack->p());
    qOverpResolution[0]->Fill((recoGlbTrack->charge()/recoGlbTrack->p())-(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->p()));
    qOverpResolution[1]->Fill((recoStaGlbTrack->charge()/recoStaGlbTrack->p())-(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->p()));

    qOverptGlbTrack[0]->Fill(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->pt());
    qOverptGlbTrack[1]->Fill(recoGlbTrack->charge()/recoGlbTrack->pt());
    qOverptGlbTrack[2]->Fill(recoStaGlbTrack->charge()/recoStaGlbTrack->pt());
    qOverptResolution[0]->Fill((recoGlbTrack->charge()/recoGlbTrack->pt())-(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->pt()));
    qOverptResolution[1]->Fill((recoStaGlbTrack->charge()/recoStaGlbTrack->pt())-(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->pt()));

  }

  if(recoMu.isTrackerMuon()) {
    LogTrace(metname)<<"[MuonRecoAnalyzer] The mu is tracker only - filling the histos";
    muReco->Fill(2);

    // get the track using only the tracker data
    reco::TrackRef recoTrack = recoMu.track();

    etaTrack->Fill(recoTrack->eta());
    thetaTrack->Fill(recoTrack->theta());
    phiTrack->Fill(recoTrack->phi());
    qOverpTrack->Fill(recoTrack->charge()/recoTrack->p());
    qOverptTrack->Fill(recoTrack->charge()/recoTrack->pt());
    
  }

  if(recoMu.isStandAloneMuon()) {
    LogTrace(metname)<<"[MuonRecoAnalyzer] The mu is STA only - filling the histos";
    muReco->Fill(3);
    
    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaTrack = recoMu.standAloneMuon();

    etaStaTrack->Fill(recoStaTrack->eta());
    thetaStaTrack->Fill(recoStaTrack->theta());
    phiStaTrack->Fill(recoStaTrack->phi());
    qOverpStaTrack->Fill(recoStaTrack->charge()/recoStaTrack->p());
    qOverptStaTrack->Fill(recoStaTrack->charge()/recoStaTrack->pt());

  }
    

}
