//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                      //
//                                    WMuNuValidator                                                                    //
//                                                                                                                      //    
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                      //      
//    Basic plots before & after cuts (without Candidate formalism)                                                     //
//    Intended for a prompt validation of samples.                                                                      // 
//                                                                                                                      //      
//    Use in combination with WMuNuValidatorMacro (in bin/WMuNuValidatorMacro.cpp)                                      //
//                                                                                                                      //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "TH1D.h"
#include <map>

class WMuNuValidator : public edm::EDFilter {
public:
  WMuNuValidator (const edm::ParameterSet &);
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void beginJob();
  virtual void endJob();
  void init_histograms();
  void fill_histogram(const char*, const double&);
private:
  bool fastOption_;
  edm::InputTag trigTag_;
  edm::InputTag muonTag_;
  edm::InputTag metTag_;
  edm::InputTag jetTag_;
  edm::InputTag vertexTag_;

  const std::vector<std::string> muonTrig_;
  double ptCut_;
  double etaMinCut_;
  double etaMaxCut_;

  bool isRelativeIso_;
  bool isCombinedIso_;
  double isoCut03_;
  double mtMin_;
  double mtMax_;
  double metMin_;
  double metMax_;
  double acopCut_;

  double dxyCut_;
  double normalizedChi2Cut_;
  int trackerHitsCut_;
                            
  int pixelHitsCut_;
  int muonHitsCut_;
  bool isAlsoTrackerMuon_;
  int nMatchesCut_;

  double ptThrForZ1_;
  double ptThrForZ2_;

  double eJetMin_;
  int nJetMax_;

  unsigned int nall;
  unsigned int nrec;
  unsigned int niso;
  unsigned int nhlt;
  unsigned int nmet;
  unsigned int nsel;
  unsigned int nid;

  std::map<std::string,TH1D*> h1_;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

using namespace edm;
using namespace std;
using namespace reco;

WMuNuValidator::WMuNuValidator( const ParameterSet & cfg ) :
      fastOption_(cfg.getUntrackedParameter<bool> ("FastOption", false)),
      // Input collections
      trigTag_(cfg.getUntrackedParameter<edm::InputTag> ("TrigTag", edm::InputTag("TriggerResults::HLT"))),
      muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("pfMet"))),
      jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("ak5PFJets"))),
      vertexTag_(cfg.getUntrackedParameter<edm::InputTag> ("VertexTag", edm::InputTag("offlinePrimaryVertices"))),

      // Main cuts 
      muonTrig_(cfg.getUntrackedParameter< std::vector<std::string> > ("MuonTrig")),  
      ptCut_(cfg.getUntrackedParameter<double>("PtCut", 25.)),
      etaMinCut_(cfg.getUntrackedParameter<double>("EtaMinCut", -2.1)),
      etaMaxCut_(cfg.getUntrackedParameter<double>("EtaMaxCut", 2.1)),
      isRelativeIso_(cfg.getUntrackedParameter<bool>("IsRelativeIso", true)),
      isCombinedIso_(cfg.getUntrackedParameter<bool>("IsCombinedIso", false)),
      isoCut03_(cfg.getUntrackedParameter<double>("IsoCut03", 0.10)),
      mtMin_(cfg.getUntrackedParameter<double>("MtMin", 50.)),
      mtMax_(cfg.getUntrackedParameter<double>("MtMax", 999.)),
      metMin_(cfg.getUntrackedParameter<double>("MetMin", -999999.)),
      metMax_(cfg.getUntrackedParameter<double>("MetMax", 999999.)),
      acopCut_(cfg.getUntrackedParameter<double>("AcopCut", 999.)),

      // Muon quality cuts
      dxyCut_(cfg.getUntrackedParameter<double>("DxyCut", 0.2)),  // dxy < 0.2
      normalizedChi2Cut_(cfg.getUntrackedParameter<double>("NormalizedChi2Cut", 10.)), // chi2/ndof < 10.
      trackerHitsCut_(cfg.getUntrackedParameter<int>("TrackerHitsCut", 11)),  // NHits > 10
      pixelHitsCut_(cfg.getUntrackedParameter<int>("PixelHitsCut", 1)),       // PixelHits > 0 
      muonHitsCut_(cfg.getUntrackedParameter<int>("MuonHitsCut", 1)),         // MuonHits > 0
      isAlsoTrackerMuon_(cfg.getUntrackedParameter<bool>("IsAlsoTrackerMuon", true)),
      nMatchesCut_(cfg.getUntrackedParameter<int>("NMatchesCut", 2)),         // At least 2 Matches 

      // Z rejection
      ptThrForZ1_(cfg.getUntrackedParameter<double>("PtThrForZ1", 20.)),
      ptThrForZ2_(cfg.getUntrackedParameter<double>("PtThrForZ2", 10.)),

      // Top rejection
      eJetMin_(cfg.getUntrackedParameter<double>("EJetMin", 999999.)),
      nJetMax_(cfg.getUntrackedParameter<int>("NJetMax", 999999))
{
}

void WMuNuValidator::beginJob() {
      nall = 0;
      nsel = 0; 

            nrec = 0;
            niso = 0;
            nhlt = 0;
            nmet = 0;
                                                                 



            init_histograms();
}

void WMuNuValidator::init_histograms() {
      edm::Service<TFileService> fs;
      TFileDirectory subDir0 = fs->mkdir("BeforeCuts");
      TFileDirectory subDir1 = fs->mkdir("LastCut");
      TFileDirectory* subDir[2]; subDir[0] = &subDir0; subDir[1] = &subDir1;

      char chname[256] = "";
      char chtitle[256] = "";
      std::string chsuffix[2] = { "_BEFORECUTS", "_LASTCUT" };

      for (int i=0; i<2; ++i) {
            snprintf(chname, 255, "PT%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Muon transverse momentum [GeV]");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,100,0.,100.);

            snprintf(chname, 255, "ETA%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Muon pseudo-rapidity");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,50,-2.51,2.51);

            snprintf(chname, 255, "DXY%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Muon transverse distance to beam spot [cm]");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,100,-0.5,0.5);

            snprintf(chname, 255, "CHI2%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Normalized Chi2, global track fit");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,100,0.,20.);

            snprintf(chname, 255, "NHITS%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Number of hits, inner track");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,40,-0.5,39.5);

            snprintf(chname, 255, "NPIXELHITS%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Number of pixel hits, inner track");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,10,-0.5,10.5);

            snprintf(chname, 255, "ValidMuonHits%s", chsuffix[i].data());
            snprintf(chtitle, 255, "number Of Valid Muon Hits");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,40,-0.5,39.5);

            snprintf(chname, 255, "TKMU%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Tracker-muon flag (for global muons)");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,2,-0.5,1.5);

            snprintf(chname, 255, "MUONMATCHES%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Number of chambers with matched segments");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,10,0,10);

            snprintf(chname, 255, "GOODEWKMU%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Quality-muon flag");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,2,-0.5,1.5);

            snprintf(chname, 255, "ISO%s", chsuffix[i].data());
            if (isRelativeIso_) {
                  if (isCombinedIso_) {
                        snprintf(chtitle, 255, "Relative (combined) isolation variable");
                  } else {
                        snprintf(chtitle, 255, "Relative (tracker) isolation variable");
                  }
                  h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle, 100, 0., 1.);
            } else {
                  if (isCombinedIso_) {
                        snprintf(chtitle, 255, "Absolute (combined) isolation variable [GeV]");
                  } else {
                        snprintf(chtitle, 255, "Absolute (tracker) isolation variable [GeV]");
                  }
                  h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle, 100, 0., 20.);
            }

            snprintf(chname, 255, "TRIG%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Trigger response (OR of Muon triggers)");
            h1_[chname]  = subDir[i]->make<TH1D>(chname,chtitle,2,-0.5,1.5);

            snprintf(chname, 255, "MT%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Transverse mass (%s) [GeV]", metTag_.label().data());
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,200,0.,200.);

            snprintf(chname, 255, "MET%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Missing transverse energy (%s) [GeV]", metTag_.label().data());
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,200,0.,200.);

            snprintf(chname, 255, "SumET%s", chsuffix[i].data());
            snprintf(chtitle, 255, "#Sigma E_{T} (%s) [GeV]", metTag_.label().data());
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,350,0.,700.);


            snprintf(chname, 255, "ACOP%s", chsuffix[i].data());
            snprintf(chtitle, 255, "MU-MET (%s) acoplanarity", metTag_.label().data());
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,10,0.,M_PI);

            snprintf(chname, 255, "NZ1%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Z rejection: number of muons above %.2f GeV", ptThrForZ1_);
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle, 10, -0.5, 9.5);

            snprintf(chname, 255, "NZ2%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Z rejection: number of muons above %.2f GeV", ptThrForZ2_);
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle, 10, -0.5, 9.5);

            snprintf(chname, 255, "NJETS%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Number of jets (%s) above %.2f GeV", jetTag_.label().data(), eJetMin_);
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,10,-0.5,9.5);

            snprintf(chname, 255, "DiMuonMass%s", chsuffix[i].data());
            snprintf(chtitle, 255, "DiMuonMass");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,200,0,200);

            snprintf(chtitle, 255, "Number of Valid Primary Vertices");
            snprintf(chname, 255, "NPVS%s", chsuffix[i].data());
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,15,-0.5,14.5);

            snprintf(chtitle, 255, "Muon Charge");
            snprintf(chname, 255, "MuonCharge%s", chsuffix[i].data());
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,3,-1.5,1.5);

      }
            h1_["PTZCUT_LASTCUT"]=  subDir[1]->make<TH1D>("PTZCUT_LASTCUT","Global pt for Muons in Z",100,0.,100.);

            h1_["NMuons_LASTCUT"] = subDir[1]->make<TH1D>("NMuons_LASTCUT", "Number of Muons in the event, after ALL cuts", 10, -0.5, 9.5);
            h1_["NMuons_BEFORECUTS"] = subDir[0]->make<TH1D>("NMuons_BEFORECUTS", "Number of Muons in the event, Before cuts", 10, -0.5, 9.5);

}

void WMuNuValidator::fill_histogram(const char* name, const double& var) {
      h1_[name]->Fill(var);
}

void WMuNuValidator::endJob() {
      double all = nall;
      double esel = nsel/all;
      LogVerbatim("") << "\n>>>>>> W SELECTION SUMMARY BEGIN >>>>>>>>>>>>>>>";
      LogVerbatim("") << "Total numer of events analyzed: " << nall << " [events]";
      LogVerbatim("") << "Total numer of events selected: " << nsel << " [events]";
      LogVerbatim("") << "Overall efficiency:             " << "(" << setprecision(4) << esel*100. <<" +/- "<< setprecision(2) << sqrt(esel*(1-esel)/all)*100. << ")%";

      if (!fastOption_) { 
        double erec = nrec/all;
        double eid  = nid/all;    
        double eiso = niso/all;
        double ehlt = nhlt/all;
        double emet = nmet/all;

        double num = nrec;
        double eff = erec;
        double err = sqrt(eff*(1-eff)/all);
        LogVerbatim("") << "Passing Pt/Eta/Quality cuts:    " << num << " [events], (" << setprecision(4) << eff*100. <<" +/- "<< setprecision(2) << err*100. << ")%";

        num = nid;
        eff = eid;
        err = sqrt(eff*(1-eff)/all);
        double effstep = 0.;
        double errstep = 0.;
        if (nrec>0) effstep = eid/erec;
        if (nrec>0) errstep = sqrt(effstep*(1-effstep)/nrec);
        LogVerbatim("") << "Passing isolation cuts:         " << num << " [events], (" << setprecision(4) << eff*100. <<" +/- "<< setprecision(2) << err*100. << ")%, to previous step: (" <<  setprecision(4) << effstep*100. << " +/- "<< setprecision(2) << errstep*100. <<")%";

        num = niso;
        eff = eiso;
        err = sqrt(eff*(1-eff)/all);
        effstep = 0.;
        errstep = 0.;
        if (nid>0) effstep = eiso/eid;
        if (nid>0) errstep = sqrt(effstep*(1-effstep)/nid);
        LogVerbatim("") << "Passing isolation cuts:         " << num << " [events], (" << setprecision(4) << eff*100. <<" +/- "<< setprecision(2) << err*100. << ")%, to previous step: (" <<  setprecision(4) << effstep*100. << " +/- "<< setprecision(2) << errstep*100. <<")%";

        num = nhlt;
        eff = ehlt;
        err = sqrt(eff*(1-eff)/all);
        effstep = 0.;
        errstep = 0.;
        if (niso>0) effstep = ehlt/eiso;
        if (niso>0) errstep = sqrt(effstep*(1-effstep)/niso);
        LogVerbatim("") << "Passing HLT criteria:           " << num << " [events], (" << setprecision(4) << eff*100. <<" +/- "<< setprecision(2) << err*100. << ")%, to previous step: (" <<  setprecision(4) << effstep*100. << " +/- "<< setprecision(2) << errstep*100. <<")%";

        num = nmet;
        eff = emet; 
        err = sqrt(eff*(1-eff)/all);
        effstep = 0.;
        errstep = 0.;
        if (nhlt>0) effstep = emet/ehlt;
        if (nhlt>0) errstep = sqrt(effstep*(1-effstep)/nhlt);
        LogVerbatim("") << "Passing MET/acoplanarity cuts:  " << num << " [events], (" << setprecision(4) << eff*100. <<" +/- "<< setprecision(2) << err*100. << ")%, to previous step: (" <<  setprecision(4) << effstep*100. << " +/- "<< setprecision(2) << errstep*100. <<")%";

        num = nsel;
        eff = esel;
        err = sqrt(eff*(1-eff)/all);
        effstep = 0.;
        errstep = 0.;
        if (nmet>0) effstep = esel/emet;
        if (nmet>0) errstep = sqrt(effstep*(1-effstep)/nmet);
        LogVerbatim("") << "Passing Z/top rejection cuts:   " << num << " [events], (" << setprecision(4) << eff*100. <<" +/- "<< setprecision(2) << err*100. << ")%, to previous step: (" <<  setprecision(4) << effstep*100. << " +/- "<< setprecision(2) << errstep*100. <<")%";
      }

      LogVerbatim("") << ">>>>>> W SELECTION SUMMARY END   >>>>>>>>>>>>>>>\n";
}

bool WMuNuValidator::filter (Event & ev, const EventSetup &) {

      // Reset global event selection flags
      bool rec_sel = false;
      bool id_sel  = false;
      bool iso_sel = false;
      bool hlt_sel = false;
      bool met_sel = false;
      bool all_sel = false;

      // Muon collection
      Handle<View<Muon> > muonCollection;
      if (!ev.getByLabel(muonTag_, muonCollection)) {
            LogError("") << ">>> Muon collection does not exist !!!";
            return false;
      }
      unsigned int muonCollectionSize = muonCollection->size();

      // Beam spot
      Handle<reco::BeamSpot> beamSpotHandle;
      if (!ev.getByLabel(InputTag("offlineBeamSpot"), beamSpotHandle)) {
            LogTrace("") << ">>> No beam spot found !!!";
            return false;
      }

      // MET
      double met_px = 0.;
      double met_py = 0.;
      Handle<View<MET> > metCollection;
                                                                 if (!ev.getByLabel(metTag_, metCollection)) {
            LogError("") << ">>> MET collection does not exist !!!";
            return false;
      }
      const MET& met = metCollection->at(0);
      met_px = met.px();
      met_py = met.py();
      double met_et = sqrt(met_px*met_px+met_py*met_py);
      LogTrace("") << ">>> MET, MET_px, MET_py: " << met_et << ", " << met_px << ", " << met_py << " [GeV]";
      fill_histogram("MET_BEFORECUTS",met_et);
      fill_histogram("SumET_BEFORECUTS",met.sumEt());

            // Vertices in the event
      Handle<View<reco::Vertex> > vertexCollection;
           if (!ev.getByLabel(vertexTag_, vertexCollection)) {
                 LogError("") << ">>> Vertex collection does not exist !!!";
                 return 0;
            }
      unsigned int vertexCollectionSize = vertexCollection->size();



      int nvvertex = 0;
      for (unsigned int i=0; i<vertexCollectionSize; i++) {
            const Vertex& vertex = vertexCollection->at(i);
            if (vertex.isValid()) nvvertex++;
      }

      fill_histogram("NPVS_BEFORECUTS",nvvertex);


      bool trigger_fired = false;

      if( muonTrig_.size() == 0){
      LogWarning("") << ">>> You are not requesting any trigger !!!";
      trigger_fired = true;
      } 
      else{
      // Trigger
      Handle<TriggerResults> triggerResults;
      if (!ev.getByLabel(trigTag_, triggerResults)) {
      LogWarning("") << ">>> TRIGGER collection does not exist !!!";
      return 0;
      }
      const edm::TriggerNames & trigNames = ev.triggerNames(*triggerResults);

      for (unsigned int i=0; i<triggerResults->size(); i++)
      {
        
        std::string trigName = trigNames.triggerName(i);
                  cout<<trigName<<"   "<<triggerResults->accept(i)<<endl;
        for (unsigned int j = 0; j < muonTrig_.size(); j++)
          {
            LogDebug("") <<"\t"<<trigName<<"   -->Trigger bit: "<<triggerResults->accept(i);
            if ( trigName == muonTrig_.at(j) && triggerResults->accept(i))
            {
              trigger_fired = true;
            }
          }
      }
      
      LogTrace("") << ">>> Trigger bit: " << trigger_fired << " for one of ( " ;
      for (unsigned int k = 0; k < muonTrig_.size(); k++)
      {
        LogTrace("") << muonTrig_.at(k) << " ";
      }
      LogTrace("") << ")";
      }     
      fill_histogram("TRIG_BEFORECUTS",trigger_fired);

      // Loop to reject/control Z->mumu is done separately
      unsigned int nmuonsForZ1 = 0;
      unsigned int nmuonsForZ2 = 0;
      bool cosmic = false;
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            double pt = mu.pt();
            double dxy = mu.globalTrack()->dxy(beamSpotHandle->position());

            if (fabs(dxy)>1) { cosmic=true; break;}

            if (pt>ptThrForZ1_) nmuonsForZ1++;
            if (pt>ptThrForZ2_) nmuonsForZ2++;

            for (unsigned int j=i; j<muonCollectionSize; j++) {
                  if (i==j) continue;
                  const Muon& mu2 = muonCollection->at(j);
                 if (mu2.isGlobalMuon() && (mu.charge()*mu2.charge()==-1) ){
                         const math::XYZTLorentzVector ZRecoGlb (mu.px()+mu2.px(), mu.py()+mu2.py() , mu.pz()+mu2.pz(), mu.p()+mu2.p());
                         fill_histogram("DiMuonMass_BEFORECUTS",ZRecoGlb.mass());
                 }
            }
      }
      if(cosmic) return 0;

      LogTrace("") << "> Z rejection: muons above " << ptThrForZ1_ << " [GeV]: " << nmuonsForZ1;
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ2_ << " [GeV]: " << nmuonsForZ2;
      fill_histogram("NZ1_BEFORECUTS",nmuonsForZ1);
      fill_histogram("NZ2_BEFORECUTS",nmuonsForZ2);
      fill_histogram("NMuons_BEFORECUTS",muonCollectionSize);

        // Jet collection 
      Handle<View<Jet> > jetCollection;
      if (!ev.getByLabel(jetTag_, jetCollection)) {
      //LogError("") << ">>> JET collection does not exist !!!";
      return 0;
      }
      unsigned int jetCollectionSize = jetCollection->size();
      int njets = 0;
      for (unsigned int i=0; i<jetCollectionSize; i++) {
            const Jet& jet = jetCollection->at(i);
                  double minDistance=99999; // This is in order to use PFJets
                  for (unsigned int i=0; i<muonCollectionSize; i++) {
                        const Muon& mu = muonCollection->at(i);
                        double distance = sqrt( (mu.eta()-jet.eta())*(mu.eta()-jet.eta()) +(mu.phi()-jet.phi())*(mu.phi()-jet.phi()) );
                        if (minDistance>distance) minDistance=distance;
                  }
                  if (minDistance<0.3) continue; // 0.3 is the isolation cone around the muon
            if (jet.et()>eJetMin_) njets++;
      }
      LogTrace("") << ">>> Total number of jets: " << jetCollectionSize;
      LogTrace("") << ">>> Number of jets above " << eJetMin_ << " [GeV]: " << njets;
      fill_histogram("NJETS_BEFORECUTS",njets);



      // Start counting
      nall++;
      bool selectZ=false;
      if (nmuonsForZ1>=1 && nmuonsForZ2>=2) selectZ=true;

      // Histograms per event shouldbe done only once, so keep track of them
      bool hlt_hist_done = false;
      bool met_hist_done = false;
      bool nz1_hist_done = false;
      bool nz2_hist_done = false;
      bool njets_hist_done = false;

      // Central W->mu nu selection criteria
      const int NFLAGS = 16;
      bool muon_sel[NFLAGS];
      bool muon4Z=false;

      for (unsigned int i=0; i<muonCollectionSize; i++) {
                                                                                               for (int j=0; j<NFLAGS; ++j) {
                  muon_sel[j] = false;
            }

            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            if (mu.globalTrack().isNull()) continue;

            LogTrace("") << "> Wsel: processing muon number " << i << "...";
            reco::TrackRef gm = mu.globalTrack();

            // Pt,eta cuts
            double pt = mu.pt();
            double eta = mu.eta();
            LogTrace("") << "\t... pt, eta: " << pt << " [GeV], " << eta;;
            if (pt>ptCut_) muon_sel[0] = true;
            if (eta>etaMinCut_ && eta<etaMaxCut_) muon_sel[1] = true;

            // d0, chi2, nhits quality cuts
            double dxy = gm->dxy(beamSpotHandle->position());
            double normalizedChi2 = gm->normalizedChi2();
            int validmuonhits=gm->hitPattern().numberOfValidMuonHits();
            int trackerHits = gm->hitPattern().numberOfValidTrackerHits();
            int pixelHits = gm->hitPattern().numberOfValidPixelHits();
            int    nMatches = mu.numberOfMatches();

            LogTrace("") << "\t... dxy, normalizedChi2, muonhits, trackerHits, pixelHits, isTrackerMuon?, nMatches: " << dxy << " [cm], " << normalizedChi2 << ", " <<validmuonhits<<" , "<< trackerHits <<" , "<< pixelHits <<  ", " << mu.isTrackerMuon()<<", "<<nMatches;

            if (fabs(dxy)<=dxyCut_) muon_sel[2] = true;
            if (normalizedChi2<=normalizedChi2Cut_) muon_sel[3] = true;
            if (validmuonhits>=muonHitsCut_) muon_sel[4] = true;
            if (trackerHits>=trackerHitsCut_) muon_sel[5] = true;
            if (pixelHits>=pixelHitsCut_) muon_sel[6] = true;
            if (mu.isTrackerMuon()) muon_sel[7] = true;
            if (nMatches>=nMatchesCut_) muon_sel[8] = true;


            fill_histogram("PT_BEFORECUTS",pt);
            fill_histogram("ETA_BEFORECUTS",eta);
            fill_histogram("DXY_BEFORECUTS",dxy);

            fill_histogram("CHI2_BEFORECUTS",normalizedChi2);
            fill_histogram("NHITS_BEFORECUTS",trackerHits);
            fill_histogram("NPIXELHITS_BEFORECUTS",pixelHits);
            fill_histogram("ValidMuonHits_BEFORECUTS",validmuonhits);
            fill_histogram("TKMU_BEFORECUTS",mu.isTrackerMuon());
            fill_histogram("MUONMATCHES_BEFORECUTS",nMatches);

            bool quality = muon_sel[4]*muon_sel[2]* muon_sel[3]* muon_sel[5]*muon_sel[6]*muon_sel[7]*muon_sel[8];
            fill_histogram("GOODEWKMU_BEFORECUTS",quality);
            fill_histogram("MuonCharge_BEFORECUTS",mu.charge());

            // Isolation cuts
            double isovar = mu.isolationR03().sumPt;
            if (isCombinedIso_) {
                  isovar += mu.isolationR03().emEt;
                  isovar += mu.isolationR03().hadEt;
            }
            if (isRelativeIso_) isovar /= pt;
            if (isovar<isoCut03_) muon_sel[9] = true;

            LogTrace("") << "\t... isolation value" << isovar <<", isolated? " << muon_sel[9];
            fill_histogram("ISO_BEFORECUTS",isovar);

            // HLT (not mtched to muon for the time being)
            if (trigger_fired) muon_sel[10] = true; 


            // For Z:
            if (pt>ptThrForZ1_ && eta>etaMinCut_ && eta<etaMaxCut_ && fabs(dxy)<dxyCut_ && quality && trigger_fired && isovar<isoCut03_) { muon4Z = true;}


            // MET/MT cuts
            double w_et = met_et+ mu.pt();
            double w_px = met_px+ mu.px();
            double w_py = met_py+mu.py();
            double massT = w_et*w_et - w_px*w_px - w_py*w_py;
            massT = (massT>0) ? sqrt(massT) : 0;

            LogTrace("") << "\t... W mass, W_et, W_px, W_py: " << massT << ", " << w_et << ", " << w_px << ", " << w_py << " [GeV]";
            if (massT>mtMin_ && massT<mtMax_) muon_sel[11] = true;
            fill_histogram("MT_BEFORECUTS",massT);
            if (met_et>metMin_ && met_et<metMax_) muon_sel[12] = true; 

            // Acoplanarity cuts
            Geom::Phi<double> deltaphi(mu.phi()-atan2(met_py,met_px));
            double acop = deltaphi.value();
            if (acop<0) acop = - acop;
            acop = M_PI - acop;
            LogTrace("") << "\t... acoplanarity: " << acop;
            if (acop<acopCut_) muon_sel[13] = true; 
            fill_histogram("ACOP_BEFORECUTS",acop);

            // Remaining flags (from global event information)
            if (nmuonsForZ1<1 || nmuonsForZ2<2) muon_sel[14] = true;
            if (njets<=nJetMax_) muon_sel[15] = true;

              // Collect necessary flags "per muon"
              int flags_passed = 0;
              bool rec_sel_this = true;
              bool id_sel_this = true;
              bool iso_sel_this = true;
              bool hlt_sel_this = true;
              bool met_sel_this = true;
              bool all_sel_this = true;
              for (int j=0; j<NFLAGS; ++j) {
                  if (muon_sel[j]) flags_passed += 1;
                  if (j<2 && !muon_sel[j]) rec_sel_this = false;
                  if (j<9 && !muon_sel[j]) id_sel_this = false;
                  if (j<10 && !muon_sel[j]) iso_sel_this = false;
                  if (j<11 && !muon_sel[j]) hlt_sel_this = false;
                  if (j<14 && !muon_sel[j]) met_sel_this = false;
                  if (!muon_sel[j]) all_sel_this = false;
              }

              // "rec" => pt,eta and quality cuts are satisfied
              if (rec_sel_this) rec_sel = true;
              // "id" => id cuts are satisfied
              if (id_sel_this) id_sel = true;
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
                        fill_histogram("PT_LASTCUT",pt);
                  if (!muon_sel[1] || flags_passed==NFLAGS)
                        fill_histogram("ETA_LASTCUT",eta);
                  if (!muon_sel[2] || flags_passed==NFLAGS)
                        fill_histogram("DXY_LASTCUT",dxy);
                  if (!muon_sel[3] || flags_passed==NFLAGS)
                        fill_histogram("CHI2_LASTCUT",normalizedChi2);
                  if (!muon_sel[4] || flags_passed==NFLAGS)
                        fill_histogram("ValidMuonHits_LASTCUT",validmuonhits);
                  if (!muon_sel[5] || flags_passed==NFLAGS)
                        fill_histogram("NHITS_LASTCUT",trackerHits);
                  if (!muon_sel[6] || flags_passed==NFLAGS)
                        fill_histogram("NPIXELHITS_LASTCUT",pixelHits);
                  if (!muon_sel[7] || flags_passed==NFLAGS)
                        fill_histogram("TKMU_LASTCUT",mu.isTrackerMuon());
                  if (!muon_sel[8] || flags_passed==NFLAGS)
                        fill_histogram("MUONMATCHES_LASTCUT",nMatches);
                  if (!quality || flags_passed==NFLAGS)
                        fill_histogram("GOODEWKMU_LASTCUT",quality);
                  if (!muon_sel[9] || flags_passed==NFLAGS)
                        fill_histogram("ISO_LASTCUT",isovar);
                  if (!muon_sel[10] || flags_passed==NFLAGS)


                        if (!hlt_hist_done) fill_histogram("TRIG_LASTCUT",trigger_fired);
                        hlt_hist_done = true;
                  if (!muon_sel[11] || flags_passed==NFLAGS)
                        fill_histogram("MT_LASTCUT",massT);
                  if (!muon_sel[12] || flags_passed==NFLAGS)
                        if (!met_hist_done) {
                        fill_histogram("MET_LASTCUT",met_et);
                        fill_histogram("SumET_LASTCUT",met.sumEt());
                        }
                        met_hist_done = true;
                  if (!muon_sel[13] || flags_passed==NFLAGS) 
                        fill_histogram("ACOP_LASTCUT",acop);
                  if (!muon_sel[14] || flags_passed==NFLAGS){ 
                        if (!nz1_hist_done) fill_histogram("NZ1_LASTCUT",nmuonsForZ1);
                        nz1_hist_done = true;
                        if (!nz2_hist_done) fill_histogram("NZ2_LASTCUT",nmuonsForZ2);
                        nz2_hist_done = true;
                  }
                  if (!muon_sel[15] || flags_passed==NFLAGS)
                        if (!njets_hist_done) fill_histogram("NJETS_LASTCUT",njets);
                        njets_hist_done = true;
                  if ( flags_passed==NFLAGS ) {
                        fill_histogram("NMuons_LASTCUT",muonCollectionSize);
                        fill_histogram("NPVS_LASTCUT",nvvertex);
                        fill_histogram("MuonCharge_LASTCUT",mu.charge());
                  }
              }

            // The cases in which the event is rejected as a Z are considered independently:

           if ( muon4Z &&  !muon_sel[14]){
                   // Plots for 2 muons       
                   bool usedMuon=false;
                   for (unsigned int j=i+1; j<muonCollectionSize; j++) {
                         const Muon& mu2 = muonCollection->at(j);
                              if (!mu2.isGlobalMuon()) continue;
                              if (mu2.charge() * mu.charge() != -1 ) continue;
                                    double pt2 = mu2.pt(); if (pt2<=ptThrForZ1_) continue;
                                    double eta2=mu2.eta(); if (eta2<=etaMinCut_ || eta2 > etaMaxCut_) continue;
                                    double isovar2 = mu2.isolationR03().sumPt;
                                    if (isCombinedIso_) {
                                          isovar2 += mu2.isolationR03().emEt;
                                          isovar2 += mu2.isolationR03().hadEt;
                                    }
                                    if (isRelativeIso_) isovar2 /= pt2;
                                    if (isovar2>=isoCut03_) continue;
                               const math::XYZTLorentzVector ZRecoGlb (mu.px()+mu2.px(), mu.py()+mu2.py() , mu.pz()+mu2.pz(), mu.p()+mu2.p());
                               fill_histogram("DiMuonMass_LASTCUT",ZRecoGlb.mass());
                               if(!usedMuon){fill_histogram("PTZCUT_LASTCUT",mu.pt()); usedMuon=true;}
                  }
            }

      }

      // Collect final flags
            if (rec_sel) nrec++;
            if (id_sel)  nid++;
            if (iso_sel) niso++;
            if (hlt_sel) nhlt++;
            if (met_sel) nmet++;

      if (all_sel) {
            nsel++;
            LogTrace("") << ">>>> Event ACCEPTED";
      } else {
            LogTrace("") << ">>>> Event REJECTED";
      }

      return all_sel;

}

#include "FWCore/Framework/interface/MakerMacros.h"

      DEFINE_FWK_MODULE( WMuNuValidator );
