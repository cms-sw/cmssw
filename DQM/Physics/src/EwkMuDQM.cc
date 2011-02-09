#include "DQM/Physics/src/EwkMuDQM.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/Common/interface/View.h"
  
using namespace edm;
using namespace std;
using namespace reco;

EwkMuDQM::EwkMuDQM( const ParameterSet & cfg ) :
      // Input collections
      trigTag_(cfg.getUntrackedParameter<edm::InputTag> ("TrigTag", edm::InputTag("TriggerResults::HLT"))),
      muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met"))),
      metIncludesMuons_(cfg.getUntrackedParameter<bool> ("METIncludesMuons", false)),
      jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("sisCone5CaloJets"))),

      // Main cuts 
      muonTrig_(cfg.getUntrackedParameter<std::string> ("MuonTrig", "HLT_Mu9")),
      ptCut_(cfg.getUntrackedParameter<double>("PtCut", 25.)),
      etaCut_(cfg.getUntrackedParameter<double>("EtaCut", 2.1)),
      isRelativeIso_(cfg.getUntrackedParameter<bool>("IsRelativeIso", true)),
      isCombinedIso_(cfg.getUntrackedParameter<bool>("IsCombinedIso", false)),
      isoCut03_(cfg.getUntrackedParameter<double>("IsoCut03", 0.1)),
      mtMin_(cfg.getUntrackedParameter<double>("MtMin", 50.)),
      mtMax_(cfg.getUntrackedParameter<double>("MtMax", 200.)),
      metMin_(cfg.getUntrackedParameter<double>("MetMin", -999999.)),
      metMax_(cfg.getUntrackedParameter<double>("MetMax", 999999.)),
      acopCut_(cfg.getUntrackedParameter<double>("AcopCut", 2.)),

      // Muon quality cuts
      dxyCut_(cfg.getUntrackedParameter<double>("DxyCut", 0.2)),
      normalizedChi2Cut_(cfg.getUntrackedParameter<double>("NormalizedChi2Cut", 10.)),
      trackerHitsCut_(cfg.getUntrackedParameter<int>("TrackerHitsCut", 11)),
      isAlsoTrackerMuon_(cfg.getUntrackedParameter<bool>("IsAlsoTrackerMuon", true)),

      // Z rejection
      ptThrForZ1_(cfg.getUntrackedParameter<double>("PtThrForZ1", 20.)),
      ptThrForZ2_(cfg.getUntrackedParameter<double>("PtThrForZ2", 10.)),

      // Top rejection
      eJetMin_(cfg.getUntrackedParameter<double>("EJetMin", 999999.)),
      nJetMax_(cfg.getUntrackedParameter<int>("NJetMax", 999999))
{
}

void EwkMuDQM::beginRun(const Run& r, const EventSetup&) {
      nall = 0;
      nsel = 0;

      nrec = 0; 
      niso = 0; 
      nhlt = 0; 
      nmet = 0;
}


void EwkMuDQM::beginJob() {
      theDbe = Service<DQMStore>().operator->();
      theDbe->setCurrentFolder("Physics/EwkMuDQM");

      init_histograms();
}

void EwkMuDQM::init_histograms() {

      char chtitle[256] = "";
      for (int i=0; i<2; ++i) {
            snprintf(chtitle, 255, "Muon transverse momentum (global muon) [GeV]");
            pt_before_ = theDbe->book1D("PT_BEFORECUTS",chtitle,100,0.,100.);
            pt_after_ = theDbe->book1D("PT_LASTCUT",chtitle,100,0.,100.);

            snprintf(chtitle, 255, "Muon pseudo-rapidity");
            eta_before_ = theDbe->book1D("ETA_BEFORECUTS",chtitle,50,-2.5,2.5);
            eta_after_ = theDbe->book1D("ETA_LASTCUT",chtitle,50,-2.5,2.5);

            snprintf(chtitle, 255, "Muon transverse distance to beam spot [cm]");
            dxy_before_ = theDbe->book1D("DXY_BEFORECUTS",chtitle,100,-0.5,0.5);
            dxy_after_ = theDbe->book1D("DXY_LASTCUT",chtitle,100,-0.5,0.5);

            snprintf(chtitle, 255, "Normalized Chi2, inner track fit");
            chi2_before_ = theDbe->book1D("CHI2_BEFORECUTS",chtitle,100,0.,100.);
            chi2_after_ = theDbe->book1D("CHI2_LASTCUT",chtitle,100,0.,100.);

            snprintf(chtitle, 255, "Number of hits, inner track");
            nhits_before_ = theDbe->book1D("NHITS_BEFORECUTS",chtitle,40,-0.5,39.5);
            nhits_after_ = theDbe->book1D("NHITS_LASTCUT",chtitle,40,-0.5,39.5);

            snprintf(chtitle, 255, "number Of Valid Muon Hits");
            muonhits_before_= theDbe->book1D("MUONHITS_BEFORECUTS",chtitle,40,-0.5,39.5);
            muonhits_after_= theDbe->book1D("MUONHITS_LASTCUT",chtitle,40,-0.5,39.5);

            snprintf(chtitle, 255, "Tracker-muon flag (for global muons)");
            tkmu_before_ = theDbe->book1D("TKMU_BEFORECUTS",chtitle,2,-0.5,1.5);
            tkmu_after_ = theDbe->book1D("TKMU_LASTCUT",chtitle,2,-0.5,1.5);

            snprintf(chtitle, 255, "Quality-muon flag");
            goodewkmuon_before_ = theDbe->book1D("GOODEWKMUON_BEFORECUTS",chtitle,2,-0.5,1.5);
            goodewkmuon_after_ = theDbe->book1D("GOODEWKMUON_LASTCUT",chtitle,2,-0.5,1.5);

            if (isRelativeIso_) {
                  if (isCombinedIso_) {
                        snprintf(chtitle, 255, "Relative (combined) isolation variable");
                  } else {
                        snprintf(chtitle, 255, "Relative (tracker) isolation variable");
                  }
                  iso_before_ = theDbe->book1D("ISO_BEFORECUTS",chtitle,100, 0., 1.);
                  iso_after_ = theDbe->book1D("ISO_LASTCUT",chtitle,100, 0., 1.);
            } else {
                  if (isCombinedIso_) {
                        snprintf(chtitle, 255, "Absolute (combined) isolation variable [GeV]");
                  } else {
                        snprintf(chtitle, 255, "Absolute (tracker) isolation variable [GeV]");
                  }
                  iso_before_ = theDbe->book1D("ISO_BEFORECUTS",chtitle,100, 0., 20.);
                  iso_after_ = theDbe->book1D("ISO_LASTCUT",chtitle,100, 0., 20.);
            }

            snprintf(chtitle, 255, "Trigger response (bit %s)", muonTrig_.data());
            trig_before_ = theDbe->book1D("TRIG_BEFORECUTS",chtitle,2,-0.5,1.5);
            trig_after_ = theDbe->book1D("TRIG_LASTCUT",chtitle,2,-0.5,1.5);

            snprintf(chtitle, 255, "Transverse mass (%s) [GeV]", metTag_.label().data());
            mt_before_ = theDbe->book1D("MT_BEFORECUTS",chtitle,150,0.,300.);
            mt_after_ = theDbe->book1D("MT_LASTCUT",chtitle,150,0.,300.);

            snprintf(chtitle, 255, "Missing transverse energy (%s) [GeV]", metTag_.label().data());
            met_before_ = theDbe->book1D("MET_BEFORECUTS",chtitle,100,0.,200.);
            met_after_ = theDbe->book1D("MET_LASTCUT",chtitle,100,0.,200.);

            snprintf(chtitle, 255, "MU-MET (%s) acoplanarity", metTag_.label().data());
            acop_before_ = theDbe->book1D("ACOP_BEFORECUTS",chtitle,50,0.,M_PI);
            acop_after_ = theDbe->book1D("ACOP_LASTCUT",chtitle,50,0.,M_PI);

            snprintf(chtitle, 255, "Z rejection: number of muons above %.2f GeV", ptThrForZ1_);
            nz1_before_ = theDbe->book1D("NZ1_BEFORECUTS",chtitle,10,-0.5,9.5);
            nz1_after_ = theDbe->book1D("NZ1_LASTCUT",chtitle,10,-0.5,9.5);

            snprintf(chtitle, 255, "Z rejection: number of muons above %.2f GeV", ptThrForZ2_);
            nz2_before_ = theDbe->book1D("NZ2_BEFORECUTS",chtitle,10,-0.5,9.5);
            nz2_after_ = theDbe->book1D("NZ2_LASTCUT",chtitle,10,-0.5,9.5);

            snprintf(chtitle, 255, "Number of jets (%s) above %.2f GeV", jetTag_.label().data(), eJetMin_);
            njets_before_ = theDbe->book1D("NJETS_BEFORECUTS",chtitle,10,-0.5,9.5);
            njets_after_ = theDbe->book1D("NJETS_LASTCUT",chtitle,10,-0.5,9.5);

            snprintf(chtitle, 255, "DiMuonMass (2 globals)");
            dimuonmass_before_= theDbe->book1D("DIMUONMASS_BEFORECUTS",chtitle,100,0,200);
            dimuonmass_after_= theDbe->book1D("DIMUONMASS_AFTERZCUTS",chtitle,100,0,200);

            snprintf(chtitle, 255, "DiMuon Mass (global pt + StandAlone pt");
            dimuonSAmass_before_= theDbe->book1D("DIMUONSTAMASS_BEFORECUTS",chtitle,100,0,200);
            dimuonSAmass_after_= theDbe->book1D("DIMUONSTAMASS_AFTERZCUTS",chtitle,100,0,200);

            snprintf(chtitle, 255, "DiMuon Mass (StandAlone pt + StandAlone pt");
            dimuonSASAmass_before_= theDbe->book1D("DIMUONSTASTAMASS_BEFORECUTS",chtitle,100,0,200); 
            dimuonSASAmass_after_= theDbe->book1D("DIMUONSTASTAMASS_AFTERZCUTS",chtitle,100,0,200);
            
            snprintf(chtitle, 255, "Global pt for Muons in Z");
            ptmuonZ_after_= theDbe->book1D("PT_AFTERZCUT",chtitle,100,0.,100.);
      }
}


void EwkMuDQM::endJob() {
}

void EwkMuDQM::endRun(const Run& r, const EventSetup&) {

}

void EwkMuDQM::analyze (const Event & ev, const EventSetup &) {
      
      // Reset global event selection flags
      bool rec_sel = false;
      bool iso_sel = false;
      bool hlt_sel = false;
      bool met_sel = false;
      bool all_sel = false;

      // Muon collection
      Handle<View<Muon> > muonCollection;
      if (!ev.getByLabel(muonTag_, muonCollection)) {
	//LogWarning("") << ">>> Muon collection does not exist !!!";
	return;
      }
      unsigned int muonCollectionSize = muonCollection->size();

      // Beam spot
      Handle<reco::BeamSpot> beamSpotHandle;
      if (!ev.getByLabel(InputTag("offlineBeamSpot"), beamSpotHandle)) {
	//LogWarning("") << ">>> No beam spot found !!!";
	return;
      }
  
      // MET
      double met_px = 0.;
      double met_py = 0.;
      Handle<View<MET> > metCollection;
      if (!ev.getByLabel(metTag_, metCollection)) {
	//LogWarning("") << ">>> MET collection does not exist !!!";
	return;
      }
      const MET& met = metCollection->at(0);
      met_px = met.px();
      met_py = met.py();
      if (!metIncludesMuons_) {
            for (unsigned int i=0; i<muonCollectionSize; i++) {
                  const Muon& mu = muonCollection->at(i);
                  if (!mu.isGlobalMuon()) continue;
                  met_px -= mu.px();
                  met_py -= mu.py();
            }
      }
      double met_et = sqrt(met_px*met_px+met_py*met_py);
      LogTrace("") << ">>> MET, MET_px, MET_py: " << met_et << ", " << met_px << ", " << met_py << " [GeV]";
      met_before_->Fill(met_et);

      // Trigger
      Handle<TriggerResults> triggerResults;
      if (!ev.getByLabel(trigTag_, triggerResults)) {
	//LogWarning("") << ">>> TRIGGER collection does not exist !!!";
	return;
      }
      const edm::TriggerNames & trigNames = ev.triggerNames(*triggerResults);
      bool trigger_fired = false;
      /*
      for (unsigned int i=0; i<triggerResults->size(); i++) {
            if (triggerResults->accept(i)) {
                  LogTrace("") << "Accept by: " << i << ", Trigger: " << trigNames.triggerName(i);
            }
      }
      */

      // the following gives error on CRAFT08 data where itrig1=19 (vector index out of range)
      /*
      int itrig1 = trigNames.triggerIndex(muonTrig_);
      if (triggerResults->accept(itrig1)) trigger_fired = true;
      */
      //suggested replacement: lm250909
      for (unsigned int i=0; i<triggerResults->size(); i++) {
        std::string trigName = trigNames.triggerName(i);
	if ( trigName == muonTrig_ && triggerResults->accept(i)) trigger_fired = true;
      }


      LogTrace("") << ">>> Trigger bit: " << trigger_fired << " (" << muonTrig_ << ")";
      trig_before_->Fill(trigger_fired);

      // Loop to reject/control Z->mumu is done separately
      unsigned int nmuonsForZ1 = 0;
      unsigned int nmuonsForZ2 = 0;
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            double pt = mu.pt();
            if (pt>ptThrForZ1_) nmuonsForZ1++;
            if (pt>ptThrForZ2_) nmuonsForZ2++;

            for (unsigned int j=0; j<muonCollectionSize; j++) {
                  if (i==j) continue;
                  const Muon& mu2 = muonCollection->at(j);
                 // Glb + Glb  
                 if (mu2.isGlobalMuon() && j>i ){
                         const math::XYZTLorentzVector ZRecoGlb (mu.px()+mu2.px(), mu.py()+mu2.py() , mu.pz()+mu2.pz(), mu.p()+mu2.p());
                         dimuonmass_before_->Fill(ZRecoGlb.mass());
                 }
                  // Glb + Standalone 
                 if (mu2.isStandAloneMuon()){
                         const math::XYZTLorentzVector ZRecoSta (mu2.outerTrack()->px()+mu.px(), mu.py()+mu.outerTrack()->py() , mu.pz()+mu2.outerTrack()->pz(), mu.p()+mu2.outerTrack()->p());
                         dimuonSAmass_before_->Fill(ZRecoSta.mass());
                 }
                  // Standalone + Standalone 
                 if (mu2.isStandAloneMuon() && j>i){
                         const math::XYZTLorentzVector ZRecoStaSta (mu2.outerTrack()->px()+mu.outerTrack()->px(), mu.outerTrack()->py()+mu.outerTrack()->py() , mu.outerTrack()->pz()+mu2.outerTrack()->pz(), mu.outerTrack()->p()+mu2.outerTrack()->p());
                         dimuonSASAmass_before_->Fill(ZRecoStaSta.mass());
                 }
            }
      
      }
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ1_ << " [GeV]: " << nmuonsForZ1;
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ2_ << " [GeV]: " << nmuonsForZ2;
      nz1_before_->Fill(nmuonsForZ1);
      nz2_before_->Fill(nmuonsForZ2);
      
      // Jet collection
      Handle<View<Jet> > jetCollection;
      if (!ev.getByLabel(jetTag_, jetCollection)) {
	//LogError("") << ">>> JET collection does not exist !!!";
	return;
      }
      unsigned int jetCollectionSize = jetCollection->size();
      int njets = 0;
      for (unsigned int i=0; i<jetCollectionSize; i++) {
            const Jet& jet = jetCollection->at(i);
            if (jet.et()>eJetMin_) njets++;
      }
      LogTrace("") << ">>> Total number of jets: " << jetCollectionSize;
      LogTrace("") << ">>> Number of jets above " << eJetMin_ << " [GeV]: " << njets;
      njets_before_->Fill(njets);

      // Start counting
      nall++;

      // Histograms per event shouldbe done only once, so keep track of them
      bool hlt_hist_done = false;
      bool met_hist_done = false;
      bool nz1_hist_done = false;
      bool nz2_hist_done = false;
      bool njets_hist_done = false;

      // Central W->mu nu selection criteria
      const int NFLAGS = 13;
      bool muon_sel[NFLAGS];
      bool muon4Z=true;

      for (unsigned int i=0; i<muonCollectionSize; i++) {
            for (int j=0; j<NFLAGS; ++j) {
                  muon_sel[j] = false;
            }

            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            if (mu.globalTrack().isNull()) continue;
            if (mu.innerTrack().isNull()) continue;

            LogTrace("") << "> Wsel: processing muon number " << i << "...";
            reco::TrackRef gm = mu.globalTrack();
            reco::TrackRef tk = mu.innerTrack();

            // Pt,eta cuts
            double pt = mu.pt();
            double eta = mu.eta();
            LogTrace("") << "\t... pt, eta: " << pt << " [GeV], " << eta;;
            if (pt>ptCut_) muon_sel[0] = true; 
            if (fabs(eta)<etaCut_) muon_sel[1] = true; 
            if (pt<ptThrForZ1_) { muon4Z = false;}

            // d0, chi2, nhits quality cuts
            double dxy = tk->dxy(beamSpotHandle->position());
            double normalizedChi2 = gm->normalizedChi2();
            double trackerHits = tk->numberOfValidHits();
            double validmuonhits=gm->hitPattern().numberOfValidMuonHits();
            LogTrace("") << "\t... dxy, normalizedChi2, trackerHits, isTrackerMuon?: " << dxy << " [cm], " << normalizedChi2 << ", " << trackerHits << ", " << mu.isTrackerMuon();
            if (fabs(dxy)<dxyCut_) muon_sel[2] = true; 
//            if (normalizedChi2<normalizedChi2Cut_) muon_sel[3] = true; 
            if (muon::isGoodMuon(mu,muon::GlobalMuonPromptTight)) muon_sel[3] = true;
            if (trackerHits>=trackerHitsCut_) muon_sel[4] = true; 
            if (mu.isTrackerMuon()) muon_sel[5] = true; 

            pt_before_->Fill(pt);
            eta_before_->Fill(eta);
            dxy_before_->Fill(dxy);
            chi2_before_->Fill(normalizedChi2);
            nhits_before_->Fill(trackerHits);
            muonhits_before_->Fill(validmuonhits);
            tkmu_before_->Fill(mu.isTrackerMuon());

            bool quality = muon_sel[4]*muon_sel[2]* muon_sel[3]* muon_sel[5];
            goodewkmuon_before_->Fill(quality);

            // Isolation cuts
            double isovar = mu.isolationR03().sumPt;
            if (isCombinedIso_) {
                  isovar += mu.isolationR03().emEt;
                  isovar += mu.isolationR03().hadEt;
            }
            if (isRelativeIso_) isovar /= pt;
            if (isovar<isoCut03_) muon_sel[6] = true; 
            if (isovar>=isoCut03_) { muon4Z = false;}

            LogTrace("") << "\t... isolation value" << isovar <<", isolated? " << muon_sel[6];
            iso_before_->Fill(isovar);


            // HLT (not mtched to muon for the time being)
            if (trigger_fired) muon_sel[7] = true; 
            else { muon4Z = false;}

            // MET/MT cuts
            double w_et = met_et+mu.pt();
            double w_px = met_px+mu.px();
            double w_py = met_py+mu.py();
            
            double massT = w_et*w_et - w_px*w_px - w_py*w_py;
            massT = (massT>0) ? sqrt(massT) : 0;

            LogTrace("") << "\t... W mass, W_et, W_px, W_py: " << massT << ", " << w_et << ", " << w_px << ", " << w_py << " [GeV]";
            if (massT>mtMin_ && massT<mtMax_) muon_sel[8] = true; 
            mt_before_->Fill(massT);
            if (met_et>metMin_ && met_et<metMax_) muon_sel[9] = true; 

            // Acoplanarity cuts
            Geom::Phi<double> deltaphi(mu.phi()-atan2(met_py,met_px));
            double acop = deltaphi.value();
            if (acop<0) acop = - acop;
            acop = M_PI - acop;
            LogTrace("") << "\t... acoplanarity: " << acop;
            if (acop<acopCut_) muon_sel[10] = true; 
            acop_before_->Fill(acop);

            // Remaining flags (from global event information)
            if (nmuonsForZ1<1 || nmuonsForZ2<2) muon_sel[11] = true; 
            if (njets<=nJetMax_) muon_sel[12] = true; 

            // Collect necessary flags "per muon"
            int flags_passed = 0;
            bool rec_sel_this = true;
            bool iso_sel_this = true;
            bool hlt_sel_this = true;
            bool met_sel_this = true;
            bool all_sel_this = true;
            for (int j=0; j<NFLAGS; ++j) {
                  if (muon_sel[j]) flags_passed += 1;
                  if (j<6 && !muon_sel[j]) rec_sel_this = false;
                  if (j<7 && !muon_sel[j]) iso_sel_this = false;
                  if (j<8 && !muon_sel[j]) hlt_sel_this = false;
                  if (j<11 && !muon_sel[j]) met_sel_this = false;
                  if (!muon_sel[j]) all_sel_this = false;
            }

            // "rec" => pt,eta and quality cuts are satisfied
            if (rec_sel_this) rec_sel = true;
            // "iso" => "rec" AND "muon is isolated"
            if (iso_sel_this) iso_sel = true;
            // "hlt" => "iso" AND "event is triggered"
            if (hlt_sel_this) hlt_sel = true;
            // "met" => "hlt" AND "MET/MT and acoplanarity cuts"
            if (met_sel_this) met_sel = true;
            // "all" => "met" AND "Z/top rejection cuts"
            if (all_sel_this) all_sel = true;

            // Do N-1 histograms now (and only once for global event quantities)
            if (flags_passed >= (NFLAGS-1)) {
                  if (!muon_sel[0] || flags_passed==NFLAGS) 
                        pt_after_->Fill(pt);
                  if (!muon_sel[1] || flags_passed==NFLAGS) 
                        eta_after_->Fill(eta);
                  if (!muon_sel[2] || flags_passed==NFLAGS) 
                        dxy_after_->Fill(dxy);
                  if (!muon_sel[3] || flags_passed==NFLAGS){ 
                        chi2_after_->Fill(normalizedChi2);
                        muonhits_after_->Fill(validmuonhits);
                  }
                  if (!muon_sel[4] || flags_passed==NFLAGS) 
                        nhits_after_->Fill(trackerHits);
                  if (!muon_sel[5] || flags_passed==NFLAGS) 
                        tkmu_after_->Fill(mu.isTrackerMuon());
                  if (!muon_sel[6] || flags_passed==NFLAGS) 
                        iso_after_->Fill(isovar);
                  if (!muon_sel[2]||!muon_sel[3] || !muon_sel[4] || !muon_sel[5] || flags_passed==NFLAGS) 
                        goodewkmuon_after_->Fill(quality);
                  if (!muon_sel[7] || flags_passed==NFLAGS) 
                        if (!hlt_hist_done) trig_after_->Fill(trigger_fired);
                        hlt_hist_done = true;
                  if (!muon_sel[8] || flags_passed==NFLAGS) 
                        mt_after_->Fill(massT);
                  if (!muon_sel[9] || flags_passed==NFLAGS) 
                        if (!met_hist_done) met_after_->Fill(met_et);
                        met_hist_done = true;
                  if (!muon_sel[10] || flags_passed==NFLAGS) 
                        acop_after_->Fill(acop);
                  if (!muon_sel[11] || flags_passed==NFLAGS) 
                        if (!nz1_hist_done) nz1_after_->Fill(nmuonsForZ1);
                        nz1_hist_done = true;
                  if (!muon_sel[11] || flags_passed==NFLAGS) 
                        if (!nz2_hist_done) nz2_after_->Fill(nmuonsForZ2);
                        nz2_hist_done = true;
                  if (!muon_sel[12] || flags_passed==NFLAGS) 
                        if (!njets_hist_done) njets_after_->Fill(njets);
                        njets_hist_done = true;
            }


            // The cases in which the event is rejected as a Z are considered independently:
            if ( muon4Z &&  !muon_sel[11]){
                   // Plots for 2 muons       
                   bool usedMuon=false;
                   for (unsigned int j=0; j<muonCollectionSize; j++) {
                         if (i==j) continue;
                         const Muon& mu2 = muonCollection->at(j);
                                    double pt2 = mu2.pt();
                                    double isovar2 = mu2.isolationR03().sumPt;
                                    if (isCombinedIso_) {
                                          isovar2 += mu2.isolationR03().emEt;
                                          isovar2 += mu2.isolationR03().hadEt;
                                    }
                                    if (isRelativeIso_) isovar2 /= pt2;

                          if (pt2<=ptThrForZ1_ || isovar2>=isoCut03_) continue;
                  
                  // Glb + Glb  
                             if (mu2.isGlobalMuon() && j>i ){
                               const math::XYZTLorentzVector ZRecoGlb (mu.px()+mu2.px(), mu.py()+mu2.py() , mu.pz()+mu2.pz(), mu.p()+mu2.p());
                               dimuonmass_after_->Fill(ZRecoGlb.mass());
                               if(!usedMuon){ptmuonZ_after_->Fill(mu.pt()); usedMuon=true;}
                             }
                  // Glb + Standalone 
                             if (mu2.isStandAloneMuon()){
                              const math::XYZTLorentzVector ZRecoSta (mu2.outerTrack()->px()+mu.px(), mu.py()+mu.outerTrack()->py() , mu.pz()+mu2.outerTrack()->pz(), mu.p()+mu2.outerTrack()->p());
                              dimuonSAmass_after_->Fill(ZRecoSta.mass());
                             }
                  // Standalone + Standalone 
                             if (mu2.isStandAloneMuon() && j>i){
                              const math::XYZTLorentzVector ZRecoStaSta (mu2.outerTrack()->px()+mu.outerTrack()->px(), mu.outerTrack()->py()+mu.outerTrack()->py() , mu.outerTrack()->pz()+mu2.outerTrack()->pz(), mu.outerTrack()->p()+mu2.outerTrack()->p());
                              dimuonSASAmass_after_->Fill(ZRecoStaSta.mass());
                             }
            }



      }




      }

      return;

}

