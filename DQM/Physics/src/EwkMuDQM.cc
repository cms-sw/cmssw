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
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h" 

#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/Common/interface/View.h"
 
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
 
using namespace edm;
using namespace std;
using namespace reco;

EwkMuDQM::EwkMuDQM( const ParameterSet & cfg ) :
      // Input collections
      trigTag_      (cfg.getUntrackedParameter<edm::InputTag> ("TrigTag", edm::InputTag("TriggerResults::HLT"))),
      muonTag_      (cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_       (cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("pfmet"))),
      jetTag_       (cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("ak5PFJets"))),
      phoTag_       (cfg.getUntrackedParameter<edm::InputTag> ("phoTag", edm::InputTag("photons"))),
      pfPhoTag_     (cfg.getUntrackedParameter<edm::InputTag> ("pfPhoTag", edm::InputTag("pfPhotonTranslator","pfPhot"))),
      vertexTag_    (cfg.getUntrackedParameter<edm::InputTag> ("VertexTag", edm::InputTag("offlinePrimaryVertices"))),
      trigPathNames_(cfg.getUntrackedParameter<std::vector <std::string> >("TrigPathNames")),           

      // Muon quality cuts
      isAlsoTrackerMuon_(cfg.getUntrackedParameter<bool>("IsAlsoTrackerMuon", true)),  // Glb muon also tracker muon 
      dxyCut_           (cfg.getUntrackedParameter<double>("DxyCut", 0.2)),            // dxy < 0.2 cm 
      normalizedChi2Cut_(cfg.getUntrackedParameter<double>("NormalizedChi2Cut", 10.)), // chi2/ndof (of global fit) <10.0
      trackerHitsCut_   (cfg.getUntrackedParameter<int>("TrackerHitsCut", 11)),        // Tracker Hits >10 
      pixelHitsCut_     (cfg.getUntrackedParameter<int>("PixelHitsCut", 1)),           // Pixel Hits >0
      muonHitsCut_      (cfg.getUntrackedParameter<int>("MuonHitsCut", 1)),            // Valid Muon Hits >0 
      nMatchesCut_      (cfg.getUntrackedParameter<int>("NMatchesCut", 2)),            // At least 2 Chambers with matches 

      // W-boson cuts 
      isRelativeIso_(cfg.getUntrackedParameter<bool>("IsRelativeIso", true)),
      isCombinedIso_(cfg.getUntrackedParameter<bool>("IsCombinedIso", false)),
      isoCut03_     (cfg.getUntrackedParameter<double>("IsoCut03", 0.1)),
      acopCut_      (cfg.getUntrackedParameter<double>("AcopCut", 999.)),
      metMin_       (cfg.getUntrackedParameter<double>("MetMin", -999999.)),
      metMax_       (cfg.getUntrackedParameter<double>("MetMax", 999999.)),
      mtMin_        (cfg.getUntrackedParameter<double>("MtMin", 50.)),
      mtMax_        (cfg.getUntrackedParameter<double>("MtMax", 200.)),
      ptCut_        (cfg.getUntrackedParameter<double>("PtCut", 20.)),
      etaCut_       (cfg.getUntrackedParameter<double>("EtaCut", 2.4)),

      // Z rejection
      ptThrForZ1_  (cfg.getUntrackedParameter<double>("PtThrForZ1", 20.)),
      ptThrForZ2_  (cfg.getUntrackedParameter<double>("PtThrForZ2", 10.)),

      // Z selection
      dimuonMassMin_(cfg.getUntrackedParameter<double>("dimuonMassMin", 80.)),
      dimuonMassMax_(cfg.getUntrackedParameter<double>("dimuonMassMax", 120.)), 

      // Top rejection
      eJetMin_     (cfg.getUntrackedParameter<double>("EJetMin", 999999.)),
      nJetMax_     (cfg.getUntrackedParameter<int>("NJetMax", 999999)), 

      // Photon cuts 
      ptThrForPhoton_(cfg.getUntrackedParameter<double>("ptThrForPhoton",5.)),
      nPhoMax_(cfg.getUntrackedParameter<int>("nPhoMax", 999999)) 
{
  isValidHltConfig_ = false;
}

void EwkMuDQM::beginRun(const Run& iRun, const EventSetup& iSet) {
      nall = 0;
      nsel = 0;
      nz   = 0;

      nrec = 0; 
      niso = 0; 
      nhlt = 0; 
      nmet = 0;

     // passed as parameter to HLTConfigProvider::init(), not yet used
     bool isConfigChanged = false;
     // isValidHltConfig_ used to short-circuit analyze() in case of problems
     isValidHltConfig_ = hltConfigProvider_.init( iRun, iSet, "HLT", isConfigChanged );

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
            pt_after_ = theDbe->book1D("PT_AFTERWCUTS",chtitle,100,0.,100.);

            snprintf(chtitle, 255, "Muon pseudo-rapidity");
            eta_before_ = theDbe->book1D("ETA_BEFORECUTS",chtitle,50,-2.5,2.5);
            eta_after_ = theDbe->book1D("ETA_AFTERWCUTS",chtitle,50,-2.5,2.5);

            snprintf(chtitle, 255, "Muon transverse distance to beam spot [cm]");
            dxy_before_ = theDbe->book1D("DXY_BEFORECUTS",chtitle,1000,-0.5,0.5);
            dxy_after_ = theDbe->book1D("DXY_AFTERWCUTS",chtitle,1000,-0.5,0.5);

            snprintf(chtitle, 255, "Quality-muon flag");
            goodewkmuon_before_ = theDbe->book1D("GOODEWKMUON_BEFORECUTS",chtitle,2,-0.5,1.5);
            goodewkmuon_after_ = theDbe->book1D("GOODEWKMUON_AFTERWCUTS",chtitle,2,-0.5,1.5);

            if (isRelativeIso_) {
                  if (isCombinedIso_) {
                        snprintf(chtitle, 255, "Relative (combined) isolation variable");
                  } else {
                        snprintf(chtitle, 255, "Relative (tracker) isolation variable");
                  }
                  iso_before_ = theDbe->book1D("ISO_BEFORECUTS",chtitle,100, 0., 1.);
                  iso_after_ = theDbe->book1D("ISO_AFTERWCUTS",chtitle,100, 0., 1.);
            } else {
                  if (isCombinedIso_) {
                        snprintf(chtitle, 255, "Absolute (combined) isolation variable [GeV]");
                  } else {
                        snprintf(chtitle, 255, "Absolute (tracker) isolation variable [GeV]");
                  }
                  iso_before_ = theDbe->book1D("ISO_BEFORECUTS",chtitle,100, 0., 20.);
                  iso_after_ = theDbe->book1D("ISO_AFTERWCUTS",chtitle,100, 0., 20.);
            }

            snprintf(chtitle, 255, "Trigger response (boolean of muon triggers)");
            trig_before_ = theDbe->book1D("TRIG_BEFORECUTS",chtitle,2,-0.5,1.5);
            trig_after_ = theDbe->book1D("TRIG_AFTERWCUTS",chtitle,2,-0.5,1.5);

            snprintf(chtitle, 255, "Transverse mass (%s) [GeV]", metTag_.label().data());
            mt_before_ = theDbe->book1D("MT_BEFORECUTS",chtitle,150,0.,300.);
            mt_after_ = theDbe->book1D("MT_AFTERWCUTS",chtitle,150,0.,300.);

            snprintf(chtitle, 255, "Missing transverse energy (%s) [GeV]", metTag_.label().data());
            met_before_ = theDbe->book1D("MET_BEFORECUTS",chtitle,100,0.,200.);
            met_after_ = theDbe->book1D("MET_AFTERWCUTS",chtitle,100,0.,200.);
            met_afterZ_ = theDbe->book1D("MET_AFTERZCUTS",chtitle,100,0.,200.);

            snprintf(chtitle, 255, "MU-MET (%s) acoplanarity", metTag_.label().data());
            acop_before_ = theDbe->book1D("ACOP_BEFORECUTS",chtitle,50,0.,M_PI);
            acop_after_ = theDbe->book1D("ACOP_AFTERWCUTS",chtitle,50,0.,M_PI);

            /* Clearing space:
            snprintf(chtitle, 255, "Z rejection: number of muons above %.2f GeV", ptThrForZ1_);
            nz1_before_ = theDbe->book1D("NZ1_BEFORECUTS",chtitle,10,-0.5,9.5);
            nz1_after_ = theDbe->book1D("NZ1_AFTERWCUTS",chtitle,10,-0.5,9.5);

            snprintf(chtitle, 255, "Z rejection: number of muons above %.2f GeV", ptThrForZ2_);
            nz2_before_ = theDbe->book1D("NZ2_BEFORECUTS",chtitle,10,-0.5,9.5);
            nz2_after_ = theDbe->book1D("NZ2_AFTERWCUTS",chtitle,10,-0.5,9.5);
            */

            snprintf(chtitle, 255, "Number of jets (%s) above %.2f GeV", jetTag_.label().data(), eJetMin_);
            njets_before_ = theDbe->book1D("NJETS_BEFORECUTS",chtitle,10,-0.5,9.5);
            njets_after_ = theDbe->book1D("NJETS_AFTERWCUTS",chtitle,10,-0.5,9.5);
            njets_afterZ_ = theDbe->book1D("NJETS_AFTERZCUTS",chtitle,10,-0.5,9.5);

            snprintf(chtitle, 255, "Leading Jet transverse momentum");
            leadingjet_pt_before_ = theDbe->book1D("LEADINGJET_PT_BEFORECUTS",chtitle,300,0.,300.);
            leadingjet_pt_after_ = theDbe->book1D("LEADINGJET_PT_AFTERWCUTS",chtitle,300,0.,300.);
            leadingjet_pt_afterZ_ = theDbe->book1D("LEADINGJET_PT_AFTERZCUTS",chtitle,300,0.,300.);

            snprintf(chtitle, 255, "Leading Jet pseudo-rapidity");
            leadingjet_eta_before_ = theDbe->book1D("LEADINGJET_ETA_BEFORECUTS",chtitle,50,-2.5,2.5);
            leadingjet_eta_after_ = theDbe->book1D("LEADINGJET_ETA_AFTERWCUTS",chtitle,50,-2.5,2.5);
            leadingjet_eta_afterZ_ = theDbe->book1D("LEADINGJET_ETA_AFTERZCUTS",chtitle,50,-2.5,2.5);


	    /**\ For Z-boson events  */

            snprintf(chtitle, 255, "Muon transverse momentum (global muon) [GeV]");
            pt1_afterZ_ = theDbe->book1D("PT1_AFTERZCUTS",chtitle,100,0.,100.);

            snprintf(chtitle, 255, "Muon pseudo-rapidity");
            eta1_afterZ_ = theDbe->book1D("ETA1_AFTERZCUTS",chtitle,50,-2.5,2.5);

            snprintf(chtitle, 255, "Muon transverse distance to beam spot [cm]");
            dxy1_afterZ_ = theDbe->book1D("DXY1_AFTERZCUTS",chtitle,1000,-0.5,0.5);

            snprintf(chtitle, 255, "Quality-muon flag");
            goodewkmuon1_afterZ_ = theDbe->book1D("GOODEWKMUON1_AFTERZCUTS",chtitle,2,-0.5,1.5);

            if (isRelativeIso_) {
                  if (isCombinedIso_) {
                        snprintf(chtitle, 255, "Relative (combined) isolation variable");
                  } else {
                        snprintf(chtitle, 255, "Relative (tracker) isolation variable");
                  }
                  iso1_afterZ_ = theDbe->book1D("ISO1_AFTERZCUTS",chtitle,100, 0., 1.);
                  iso2_afterZ_ = theDbe->book1D("ISO2_AFTERZCUTS",chtitle,100, 0., 1.);
            } else {
                  if (isCombinedIso_) {
                        snprintf(chtitle, 255, "Absolute (combined) isolation variable [GeV]");
                  } else {
                        snprintf(chtitle, 255, "Absolute (tracker) isolation variable [GeV]");
                  }
                  iso1_afterZ_ = theDbe->book1D("ISO1_AFTERZCUTS",chtitle,100, 0., 20.);
                  iso2_afterZ_ = theDbe->book1D("ISO2_AFTERZCUTS",chtitle,100, 0., 20.);
            }

            snprintf(chtitle, 255, "Muon transverse momentum (global muon) [GeV]");
            pt2_afterZ_ = theDbe->book1D("PT2_AFTERZCUTS",chtitle,100,0.,100.);

            snprintf(chtitle, 255, "Muon pseudo-rapidity");
            eta2_afterZ_ = theDbe->book1D("ETA2_AFTERZCUTS",chtitle,50,-2.5,2.5);

            snprintf(chtitle, 255, "Muon transverse distance to beam spot [cm]");
            dxy2_afterZ_ = theDbe->book1D("DXY2_AFTERZCUTS",chtitle,1000,-0.5,0.5);

            snprintf(chtitle, 255, "Quality-muon flag");
            goodewkmuon2_afterZ_ = theDbe->book1D("GOODEWKMUON2_AFTERZCUTS",chtitle,2,-0.5,1.5);

            snprintf(chtitle, 255, "Trigger response (boolean of muon triggers)");
            ztrig_afterZ_ = theDbe->book1D("ZTRIG_AFTERZCUTS",chtitle,2,-0.5,1.5); 

            snprintf(chtitle, 255, "DiMuonMass (2 globals)");
            dimuonmass_before_= theDbe->book1D("DIMUONMASS_BEFORECUTS",chtitle,100,0,200);
            dimuonmass_afterZ_= theDbe->book1D("DIMUONMASS_AFTERZCUTS",chtitle,100,0,200);

            snprintf(chtitle, 255, "Number of Valid Primary Vertices");
            npvs_before_ = theDbe->book1D("NPVs_BEFORECUTS",chtitle,30,-0.5,29.5);
            npvs_after_ = theDbe->book1D("NPVs_AFTERZCUTS",chtitle,30,-0.5,29.5);
            npvs_afterZ_ = theDbe->book1D("NPVs_AFTERWCUTS",chtitle,30,-0.5,29.5);

            snprintf(chtitle, 255, "Muon Charge");
            muoncharge_before_ = theDbe->book1D("MUONCHARGE_BEFORECUTS",chtitle,3,-1.5,1.5);
            muoncharge_after_ = theDbe->book1D("MUONCHARGE_AFTERZCUTS",chtitle,3,-1.5,1.5);
            muoncharge_afterZ_ = theDbe->book1D("MUONCHARGE_AFTERWCUTS",chtitle,3,-1.5,1.5);

            // Adding these to replace the NZ ones (more useful, since they are more general?)     
            snprintf(chtitle, 255, "Number of muons in the event");
            nmuons_ = theDbe->book1D("NMuons",chtitle,10,-0.5,9.5);
            snprintf(chtitle, 255, "Number of muons passing the quality criteria");
            ngoodmuons_ = theDbe->book1D("NGoodMuons",chtitle,10,-0.5,9.5);

	    snprintf(chtitle, 255, "Number of photons in the event");
	    nph_ = theDbe->book1D("nph",chtitle,20,0.,20.); 
	    snprintf(chtitle, 255, "Number of PF photons in the event");
	    npfph_ = theDbe->book1D("npfph",chtitle,20,0.,20.); 
            snprintf(chtitle, 255, "Photon transverse momentum [GeV]");
            phPt_ = theDbe->book1D("phPt",chtitle,1000,0.,1000.);
            snprintf(chtitle, 255, "PF Photon transverse momentum [GeV]");
            pfphPt_ = theDbe->book1D("pfphPt",chtitle,1000,0.,1000.); 
            snprintf(chtitle, 255, "Photon pseudorapidity");
            phEta_ = theDbe->book1D("phEta",chtitle,100,-2.5,2.5); 
            snprintf(chtitle, 255, "PF Photon pseudorapidity");
            pfphEta_ = theDbe->book1D("pfphEta",chtitle,100,-2.5,2.5); 

      }
}


void EwkMuDQM::endJob() {
}

void EwkMuDQM::endRun(const Run& r, const EventSetup& iSet) {

}

void EwkMuDQM::analyze (const Event & ev, const EventSetup & iSet) {
      
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


      // Loop to reject/control Z->mumu is done separately
      unsigned int nmuonsForZ1 = 0;
      unsigned int nmuonsForZ2 = 0;
      bool cosmic = false;
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            double pt = mu.pt();
            double dxy = mu.innerTrack()->dxy(beamSpotHandle->position());

            if (fabs(dxy)>1) { cosmic=true; break;} 

            if (pt>ptThrForZ1_) nmuonsForZ1++;
            if (pt>ptThrForZ2_) nmuonsForZ2++;

            for (unsigned int j=i+1; j<muonCollectionSize; j++) {
                 const Muon& mu2 = muonCollection->at(j);
                 if (mu2.isGlobalMuon() && (mu.charge()*mu2.charge()==-1) ){
                         const math::XYZTLorentzVector ZRecoGlb (mu.px()+mu2.px(), mu.py()+mu2.py() , mu.pz()+mu2.pz(), mu.p()+mu2.p());
                         dimuonmass_before_->Fill(ZRecoGlb.mass());
                 }
            }
      }
     if(cosmic) return;

      LogTrace("") << "> Z rejection: muons above " << ptThrForZ1_ << " [GeV]: " << nmuonsForZ1;
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ2_ << " [GeV]: " << nmuonsForZ2;
      /*nz1_before_->Fill(nmuonsForZ1);
      nz2_before_->Fill(nmuonsForZ2);
      */

      // MET
      Handle<View<MET> > metCollection;
      if (!ev.getByLabel(metTag_, metCollection)) {
	//LogWarning("") << ">>> MET collection does not exist !!!";
	return;
      }
      const MET& met = metCollection->at(0);
      double met_et = met.pt();
      LogTrace("") << ">>> MET, MET_px, MET_py: " << met_et << ", " << met.px() << ", " << met.py() << " [GeV]";
      met_before_->Fill(met_et);

      // Vertices in the event
      Handle<View<reco::Vertex> > vertexCollection;
           if (!ev.getByLabel(vertexTag_, vertexCollection)) {
                 LogError("") << ">>> Vertex collection does not exist !!!";
                 return;
            }
      unsigned int vertexCollectionSize = vertexCollection->size();

      

      int nvvertex = 0;
      for (unsigned int i=0; i<vertexCollectionSize; i++) {
            const Vertex& vertex = vertexCollection->at(i);
            if (vertex.isValid()) nvvertex++;
      }

      npvs_before_->Fill(nvvertex);

      bool trigger_fired = false;
      Handle<TriggerResults> triggerResults;
      if (!ev.getByLabel(trigTag_, triggerResults)) {
	//LogWarning("") << ">>> TRIGGER collection does not exist !!!";
	return;
      }
      const edm::TriggerNames & trigNames = ev.triggerNames(*triggerResults);
      //  LogWarning("")<<"Loop over triggers";


      for (unsigned int i=0; i<triggerResults->size(); i++)
      {
              const std::string trigName = trigNames.triggerName(i);

              bool found=false; 
              for(unsigned int index=0; index<trigPathNames_.size() && found==false; index++) {
                   size_t trigPath = trigName.find(trigPathNames_[index]); // 0 if found, pos if not
                   if (trigPath==0) found=true;
              }
              if(!found) {continue;}
              
              bool prescaled=false;    
              for (unsigned int ps= 0; ps<  hltConfigProvider_.prescaleSize(); ps++){
                  const unsigned int prescaleValue = hltConfigProvider_.prescaleValue(ps, trigName) ;
                  if (prescaleValue != 1) prescaled =true;
              }
            
              if( triggerResults->accept(i) && !prescaled){   trigger_fired=true;}
                        // LogWarning("")<<"TrigNo: "<<i<<"  "<<found<<"  "<<trigName<<" ---> FIRED";}
      }     
      trig_before_->Fill(trigger_fired);

      // Jet collection
      Handle<View<Jet> > jetCollection;
      if (!ev.getByLabel(jetTag_, jetCollection)) {
	//LogError("") << ">>> JET collection does not exist !!!";
	return;
      }
      unsigned int jetCollectionSize = jetCollection->size();
      int njets = 0; int LEADJET=-1;  double max_pt=0;
      for (unsigned int i=0; i<jetCollectionSize; i++) {
            const Jet& jet = jetCollection->at(i);
                  double minDistance=99999; // This is in order to use PFJets
                  for (unsigned int j=0; j<muonCollectionSize; j++) {
                        const Muon& mu = muonCollection->at(j);
                        double distance = sqrt( (mu.eta()-jet.eta())*(mu.eta()-jet.eta()) +(mu.phi()-jet.phi())*(mu.phi()-jet.phi()) );      
                        if (minDistance>distance) minDistance=distance;
                  }
                  if (minDistance<0.3) continue; // 0.3 is the isolation cone around the muon
            if(jet.et()>max_pt) { LEADJET=i; max_pt=jet.et();}
            if (jet.et()>eJetMin_) {njets++;}
      }


      LogTrace("") << ">>> Total number of jets: " << jetCollectionSize;
      LogTrace("") << ">>> Number of jets above " << eJetMin_ << " [GeV]: " << njets;
      njets_before_->Fill(njets);
      double lead_jet_pt=-1;
      double lead_jet_eta=-100;
      if(LEADJET!=-1){
      const Jet& leadJet = jetCollection->at(LEADJET);
      leadingjet_pt_before_->Fill(leadJet.pt());
      leadingjet_eta_before_->Fill(leadJet.eta());
      lead_jet_pt=leadJet.pt();
      lead_jet_eta=leadJet.eta();
      }
      //Photon Collection
      Handle<View<Photon> > photonCollection;
      if(!ev.getByLabel(phoTag_,photonCollection)){
      //LogError("")
      return;
      }
      unsigned int ngam=0;
      
      for (unsigned int i=0; i<photonCollection->size(); i++){
      	const Photon &ph = photonCollection->at(i);
      	double photonPt = ph.pt();
      	if (photonPt> ptThrForPhoton_) ngam++;
	phPt_->Fill(photonPt); 
      	}
      nph_->Fill(ngam); 
      LogTrace("") << " >>> N photons " << ngam << std::endl;

      nmuons_->Fill(muonCollectionSize);

      ////PF Photon Collection
      //Handle<View<Photon> > pfPhotonCollection;
      ////Handle<View<pfPhot> > pfPhotonCollection;
      //if(!ev.getByLabel(pfPhoTag_,pfPhotonCollection)){
      ////LogError("")
      //return;
      //}
      //unsigned int npfgam=0;
      //
      //for (unsigned int i=0; i<pfPhotonCollection->size(); i++){
      //	const Photon &ph = pfPhotonCollection->at(i);
      //	double photonPt = ph.pt();
      //	if (photonPt> ptThrForPhoton_) npfgam++;
      //  pfphPt_->Fill(photonPt); 
      //	}
      //npfph_->Fill(npfgam); 
      //LogTrace("") << " >>> N PF photons " << npfgam << std::endl;

      // Start counting
      nall++;

      // Histograms per event should be done only once, so keep track of them
      bool hlt_hist_done = false;
      bool zhlt_hist_done = false;
      bool zjets_hist_done = false;
      bool zfullsel_hist_done = false;
      bool met_hist_done = false;
      //bool nz1_hist_done = false;
      //bool nz2_hist_done = false;
      bool njets_hist_done = false;
      bool wfullsel_hist_done = false;

      // Central W->mu nu selection criteria
      const int NFLAGS = 11;
      bool muon_sel[NFLAGS];
      const int NFLAGSZ = 13;
      bool zmuon_sel[NFLAGSZ];
      bool muon4Z=false;

      double number_of_muons=0;
      double number_of_goodMuons=0;


      for (unsigned int i=0; i<muonCollectionSize; i++) {
            for (int j=0; j<NFLAGS; ++j) {
                  muon_sel[j] = false;
            }

            number_of_muons++;

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

            double charge=mu.charge();

            // d0, chi2, nhits quality cuts
            double dxy = gm->dxy(beamSpotHandle->position());
            double normalizedChi2 = gm->normalizedChi2();
            double trackerHits = tk->hitPattern().numberOfValidTrackerHits();
            int pixelHits = tk->hitPattern().numberOfValidPixelHits();
            int muonHits = gm->hitPattern().numberOfValidMuonHits();
            int nMatches = mu.numberOfMatches();

            LogTrace("") << "\t... dxy, normalizedChi2, trackerHits, isTrackerMuon?: " << dxy << " [cm], " << normalizedChi2 << ", " << trackerHits << ", " << mu.isTrackerMuon();
            if (fabs(dxy)<dxyCut_) muon_sel[2] = true; 

            bool quality=true;
            
            if (normalizedChi2>normalizedChi2Cut_) quality =false; 
            if (trackerHits<trackerHitsCut_) quality =false;
            if (pixelHits<pixelHitsCut_) quality =false;
            if (muonHits<muonHitsCut_) quality=false;;
            if (!mu.isTrackerMuon()) quality=false;
            if (nMatches<nMatchesCut_) quality=false;
            muon_sel[3]=quality;
            if(quality) number_of_goodMuons++;

            pt_before_->Fill(pt);
            eta_before_->Fill(eta);
            dxy_before_->Fill(dxy);
            muoncharge_before_->Fill(charge);
            goodewkmuon_before_->Fill(quality);

            // Isolation cuts
            double isovar = mu.isolationR03().sumPt;
            if (isCombinedIso_) {
                  isovar += mu.isolationR03().emEt;
                  isovar += mu.isolationR03().hadEt;
            }
            if (isRelativeIso_) isovar /= pt;
            if (isovar<isoCut03_) muon_sel[4] = true; 

            LogTrace("") << "\t... isolation value" << isovar <<", isolated? " << muon_sel[6];
            iso_before_->Fill(isovar);


            // HLT (not mtched to muon for the time being)
            if (trigger_fired) muon_sel[5] = true; 

            // For Z:
            if (pt>ptThrForZ1_ && fabs(eta)<etaCut_ && fabs(dxy)<dxyCut_ && quality && trigger_fired && isovar<isoCut03_) { muon4Z = true;}


            // MET/MT cuts
            double w_et = met_et+mu.pt();
            double w_px = met.px()+mu.px();
            double w_py = met.py()+mu.py();
            
            double massT = w_et*w_et - w_px*w_px - w_py*w_py;
            massT = (massT>0) ? sqrt(massT) : 0;

            LogTrace("") << "\t... W mass, W_et, W_px, W_py: " << massT << ", " << w_et << ", " << w_px << ", " << w_py << " [GeV]";
            if (massT>mtMin_ && massT<mtMax_) muon_sel[6] = true; 
            mt_before_->Fill(massT);
            if (met_et>metMin_ && met_et<metMax_) muon_sel[7] = true; 

            // Acoplanarity cuts
            Geom::Phi<double> deltaphi(mu.phi()-atan2(met.py(),met.px()));
            double acop = deltaphi.value();
            if (acop<0) acop = - acop;
            acop = M_PI - acop;
            LogTrace("") << "\t... acoplanarity: " << acop;
            if (acop<acopCut_) muon_sel[8] = true; 
            acop_before_->Fill(acop);

            // Remaining flags (from global event information)
            if (nmuonsForZ1<1 || nmuonsForZ2<2) muon_sel[9] = true; 
            if (njets<=nJetMax_) muon_sel[10] = true; 

            // Collect necessary flags "per muon"
            int flags_passed = 0;
            bool rec_sel_this = true;
            bool iso_sel_this = true;
            bool hlt_sel_this = true;
            bool met_sel_this = true;
            bool all_sel_this = true;
            for (int j=0; j<NFLAGS; ++j) {
                  if (muon_sel[j]) flags_passed += 1;
                  if (j<4 && !muon_sel[j]) rec_sel_this = false;
                  if (j<5 && !muon_sel[j]) iso_sel_this = false;
                  if (j<6 && !muon_sel[j]) hlt_sel_this = false;
                  if (j<8 && !muon_sel[j]) met_sel_this = false;
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
                  if (!muon_sel[3] || flags_passed==NFLAGS)
                        goodewkmuon_after_->Fill(quality);
                  if (!muon_sel[4] || flags_passed==NFLAGS) 
                        iso_after_->Fill(isovar);
                  if (!muon_sel[5] || flags_passed==NFLAGS) 
                        if (!hlt_hist_done) trig_after_->Fill(trigger_fired);
                        hlt_hist_done = true;
                  if (!muon_sel[6] || flags_passed==NFLAGS) 
                        mt_after_->Fill(massT);
                  if (!muon_sel[7] || flags_passed==NFLAGS) 
                        if (!met_hist_done) met_after_->Fill(met_et);
                        met_hist_done = true;
                  if (!muon_sel[8] || flags_passed==NFLAGS) 
                        acop_after_->Fill(acop);
                  /* Clearing some space
                  if (!muon_sel[9] || flags_passed==NFLAGS) 
                        if (!nz1_hist_done) nz1_after_->Fill(nmuonsForZ1);
                        nz1_hist_done = true;
                  if (!muon_sel[9] || flags_passed==NFLAGS) 
                        if (!nz2_hist_done) nz2_after_->Fill(nmuonsForZ2);
                        nz2_hist_done = true;
                  */
                  if (!muon_sel[10] || flags_passed==NFLAGS) { 
                        if (!njets_hist_done) {
                                    njets_after_->Fill(njets);
                                    leadingjet_pt_after_->Fill(lead_jet_pt);
                                    leadingjet_eta_after_->Fill(lead_jet_eta);
                        }
                        njets_hist_done = true;
                  if( flags_passed==NFLAGS ) {
                        if (!wfullsel_hist_done){
                        npvs_after_->Fill(nvvertex);
                        muoncharge_after_->Fill(charge);
                        }
                        wfullsel_hist_done=true;    
                  } 
                   
            }


            // The cases in which the event is rejected as a Z are considered independently:
            if ( muon4Z &&  !muon_sel[9]){

                   // Plots for 2 muons       
                   for (unsigned int j=i+1; j<muonCollectionSize; j++) {

                         for (int ij=0; ij<NFLAGSZ; ++ij) {
                               zmuon_sel[ij] = false;
                         }

		         for (int ji=0; ji<5; ++ji ) {
		           zmuon_sel[ji] = muon_sel[ji];
		         }

                         const Muon& mu2 = muonCollection->at(j);
                              if (!mu2.isGlobalMuon()) continue;
                              if (mu2.charge() * charge != -1 ) continue;
			            reco::TrackRef gm2 = mu2.globalTrack();
				    reco::TrackRef tk2 = mu2.innerTrack();
                                    double pt2 = mu2.pt(); if (pt2>ptThrForZ1_) zmuon_sel[5] = true;
                                    double eta2=mu2.eta(); if (fabs(eta2)<etaCut_) zmuon_sel[6] = true;
				    double dxy2 = gm2->dxy(beamSpotHandle->position()); if (fabs(dxy2)<dxyCut_) zmuon_sel[7] = true;
                                    double normalizedChi22 = gm2->normalizedChi2();
                                    double trackerHits2    = tk2->hitPattern().numberOfValidTrackerHits();
                                    int    pixelHits2      = tk2->hitPattern().numberOfValidPixelHits();
                                    int    muonHits2       = gm2->hitPattern().numberOfValidMuonHits();
                                    int    nMatches2       = mu2.numberOfMatches();
				    bool quality2=true;
                                    if (normalizedChi22>normalizedChi2Cut_) quality2 = false; 
                                    if (trackerHits2<trackerHitsCut_)       quality2 = false;
                                    if (pixelHits2<pixelHitsCut_)           quality2 = false;
                                    if (muonHits2<muonHitsCut_)             quality2 = false;
                                    if (!mu2.isTrackerMuon())               quality2 = false;
                                    if (nMatches2<nMatchesCut_)             quality2 = false;
                                    zmuon_sel[8]=quality2;
                                    double isovar2 = mu2.isolationR03().sumPt; 
                                    if (isCombinedIso_) {
                                          isovar2 += mu2.isolationR03().emEt;
                                          isovar2 += mu2.isolationR03().hadEt;
                                    }
                                    if (isRelativeIso_) isovar2 /= pt2;
                                    if (isovar2<isoCut03_) zmuon_sel[9] = true;
				    if (trigger_fired) zmuon_sel[10] = true; 
                               const math::XYZTLorentzVector ZRecoGlb (mu.px()+mu2.px(), mu.py()+mu2.py() , mu.pz()+mu2.pz(), mu.p()+mu2.p());
			       if (ZRecoGlb.mass()>dimuonMassMin_ && ZRecoGlb.mass()<dimuonMassMax_) zmuon_sel[11] = true; 

					//jet flag
					if (njets <=nJetMax_) zmuon_sel[12] = true; 

                               // start filling histos: N-1 plots
			       int  flags_passed_z = 0;
			       bool rec1_sel_z      = true;
			       bool iso1_sel_z      = true;
			       bool rec2_sel_z      = true;
			       bool iso2_sel_z      = true;
			       bool hlt_sel_z      = true;
			       bool zmu_sel_z      = true;
			       bool all_sel_z      = true;
			       for (int jj=0; jj<NFLAGSZ; ++jj) {
				       if (zmuon_sel[jj]) ++flags_passed_z ;
				       if (jj<4 && !zmuon_sel[jj]) rec1_sel_z = false;
				       if (jj<5 && !zmuon_sel[jj]) iso1_sel_z = false;
				       if (jj<9 && !zmuon_sel[jj]) rec2_sel_z = false;
				       if (jj<10&& !zmuon_sel[jj]) iso2_sel_z = false;
				       if (jj<11&& !zmuon_sel[jj]) hlt_sel_z  = false;
				       if (jj<12&& !zmuon_sel[jj]) zmu_sel_z  = false;
				       if (!zmuon_sel[jj]) all_sel_z  = false; 

			       }

			       if (flags_passed_z >= (NFLAGSZ-1)) {
				       if (!zmuon_sel[0]  || flags_passed_z==NFLAGSZ) {pt1_afterZ_->Fill(pt); } 
				       if (!zmuon_sel[1]  || flags_passed_z==NFLAGSZ) {eta1_afterZ_->Fill(eta); }  
				       if (!zmuon_sel[2]  || flags_passed_z==NFLAGSZ) {dxy1_afterZ_->Fill(dxy); }  
				       if (!zmuon_sel[3]  || flags_passed_z==NFLAGSZ) {goodewkmuon1_afterZ_->Fill(quality); }  
				       if (!zmuon_sel[4]  || flags_passed_z==NFLAGSZ) {iso1_afterZ_->Fill(isovar); }  
				       if (!zmuon_sel[5]  || flags_passed_z==NFLAGSZ) { pt2_afterZ_->Fill(pt2); }  
				       if (!zmuon_sel[6]  || flags_passed_z==NFLAGSZ) { eta2_afterZ_->Fill(eta2); }  
				       if (!zmuon_sel[7]  || flags_passed_z==NFLAGSZ) {dxy2_afterZ_->Fill(dxy2); }  
				       if (!zmuon_sel[8]  || flags_passed_z==NFLAGSZ) {goodewkmuon2_afterZ_->Fill(quality2); }  
				       if (!zmuon_sel[9]  || flags_passed_z==NFLAGSZ) {iso2_afterZ_->Fill(isovar2); }   
				       if (!zmuon_sel[10] || flags_passed_z==NFLAGSZ) { 
                                         if (!zhlt_hist_done) ztrig_afterZ_->Fill(trigger_fired); 
                                          zhlt_hist_done = true; 
				       }  
				       if (!zmuon_sel[11] || flags_passed_z==NFLAGSZ) {dimuonmass_afterZ_->Fill(ZRecoGlb.mass()); } 
						 if (!zmuon_sel[12] || flags_passed_z==NFLAGSZ ){
                                                            if(!zjets_hist_done){
                                                            njets_afterZ_->Fill(njets);
                                                            leadingjet_pt_afterZ_->Fill(lead_jet_pt);
                                                            leadingjet_eta_afterZ_->Fill(lead_jet_eta);
                                                            }
                                                            zjets_hist_done=true;
                                    }
                               if(flags_passed_z==NFLAGSZ) {met_afterZ_->Fill(met_et);
                                          if(!zfullsel_hist_done){
                                          npvs_after_->Fill(nvvertex);
                                          muoncharge_after_->Fill(charge);
                                          }
                                          zfullsel_hist_done=true;
                               }     
                  }

			       }

                  }
            }

      }

      //nmuons_->Fill(number_of_muons);
      //nmuons_->Fill(muonCollectionSize);
      ngoodmuons_->Fill(number_of_goodMuons);

      return;

}

