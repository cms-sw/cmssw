//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                      //
//                                    WMuNuSelector based on WMuNuCandidates                                            //
//                                                                                                                      //    
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                      //      
//    Filter of WMuNuCandidates for Analysis                                                                            //
//    --> From a WMuNuCandidate collection                                                                              //
//    --> Pre-Selection of events based in event cuts (trigger, Z rejection, ttbar rejection)                           //
//    --> The Ws are selected from the highest pt muon in the event (applying the standard WMuNu Selection cuts)        //
//                                                                                                                      //
//    Optionally, plots selection variables sequentially after cuts,                                                    //
//    and a 2D histograms for background determination.                                                                 //
//                                                                                                                      //      
//    A NTuple with the final variables of the analysis (after trigger, jets, Z veto and MuonID preselection) has been  //
//    for the final analysis. For usage in bare root, load WTuple.h                                                     //
//                                                                                                                      //
//    For basic plots before & after cuts (without Candidate formalism), use WMuNuValidator.cc                          //
//                                                                                                                      //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TNtuple.h"

class WMuNuSelector : public edm::EDFilter {

public:
  WMuNuSelector (const edm::ParameterSet &);
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void beginJob();
  virtual void endJob();
  void init_histograms();
private:
  bool plotHistograms_;
  bool saveNTuple_;

  edm::InputTag trigTag_;
  edm::InputTag muonTag_;
  edm::InputTag jetTag_;
  edm::InputTag WMuNuCollectionTag_;
  edm::InputTag vertexTag_;

  const std::vector<std::string> muonTrig_;
  double ptThrForZ1_;
  double ptThrForZ2_;
  double eJetMin_;
  int nJetMax_;
  int NVertexMin_;
  int NVertexMax_;

  double ptCut_;
  double etaCut_;
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
      
  int selectByCharge_;

  double nall; 
  double ntrig, npresel;
  double nsel;
  double ncharge;
  double neta,nkin, nid,nacop,niso,nmass;
  double ncands,nZ;


  std::map<std::string,TH1D*> h1_;
  std::map<std::string,TH2D*> h2_;
  std::map<std::string,TNtuple*> hn_;

};
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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

#include "AnalysisDataFormats/EWK/interface/WMuNuCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


#include "TAxis.h"

  
using namespace edm;
using namespace std;
using namespace reco;

WMuNuSelector::WMuNuSelector( const ParameterSet & cfg ) :
      // Fast selection (no histograms)
      plotHistograms_(cfg.getUntrackedParameter<bool> ("plotHistograms", false)),
      saveNTuple_(cfg.getUntrackedParameter<bool> ("saveNTuple", false)),

      // Input collections
      trigTag_(cfg.getUntrackedParameter<edm::InputTag> ("TrigTag", edm::InputTag("TriggerResults::HLT"))),
      muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("sisCone5CaloJets"))),
      WMuNuCollectionTag_(cfg.getUntrackedParameter<edm::InputTag> ("WMuNuCollectionTag", edm::InputTag("WMuNus"))),
      vertexTag_(cfg.getUntrackedParameter<edm::InputTag> ("VertexTag", edm::InputTag("offlinePrimaryVertices"))), 


      // Preselection cuts 
      muonTrig_(cfg.getUntrackedParameter<std::vector <std::string> >("MuonTrig")),
      ptThrForZ1_(cfg.getUntrackedParameter<double>("PtThrForZ1", 20.)),
      ptThrForZ2_(cfg.getUntrackedParameter<double>("PtThrForZ2", 10.)),
      eJetMin_(cfg.getUntrackedParameter<double>("EJetMin", 999999.)),
      nJetMax_(cfg.getUntrackedParameter<int>("NJetMax", 999999)),
      NVertexMin_(cfg.getUntrackedParameter<int>("MinVertices", 0)),
      NVertexMax_(cfg.getUntrackedParameter<int>("MaxVertices", 999)),

      // MaiB cuts 
      ptCut_(cfg.getUntrackedParameter<double>("PtCut", 25.)),
      etaCut_(cfg.getUntrackedParameter<double>("EtaCut", 2.1)),
      isRelativeIso_(cfg.getUntrackedParameter<bool>("IsRelativeIso", true)),
      isCombinedIso_(cfg.getUntrackedParameter<bool>("IsCombinedIso", true)),
      isoCut03_(cfg.getUntrackedParameter<double>("IsoCut03", 0.10)),
      mtMin_(cfg.getUntrackedParameter<double>("MtMin", 50.)),
      mtMax_(cfg.getUntrackedParameter<double>("MtMax", 9999999.)),
      metMin_(cfg.getUntrackedParameter<double>("MetMin", -999999.)),
      metMax_(cfg.getUntrackedParameter<double>("MetMax", 999999.)),
      acopCut_(cfg.getUntrackedParameter<double>("AcopCut", 999.)),   

      // Muon quality cuts
      dxyCut_(cfg.getUntrackedParameter<double>("DxyCut", 0.2)),   // dxy < 0.2 cm 
      normalizedChi2Cut_(cfg.getUntrackedParameter<double>("NormalizedChi2Cut", 10.)), // chi2/ndof (of global fit) <10.0
      trackerHitsCut_(cfg.getUntrackedParameter<int>("TrackerHitsCut", 11)),  // Tracker Hits >10 
      pixelHitsCut_(cfg.getUntrackedParameter<int>("PixelHitsCut", 1)), // Pixel Hits >0
      muonHitsCut_(cfg.getUntrackedParameter<int>("MuonHitsCut", 1)),  // Valid Muon Hits >0 
      isAlsoTrackerMuon_(cfg.getUntrackedParameter<bool>("IsAlsoTrackerMuon", true)),
      nMatchesCut_(cfg.getUntrackedParameter<int>("NMatchesCut", 2)), // At least 2 Chambers with matches 


      // W+/W- Selection
      selectByCharge_(cfg.getUntrackedParameter<int>("SelectByCharge", 0))
{
}

void WMuNuSelector::beginJob() {
      nall = 0;
      ntrig=0;
      npresel=0;
      ncharge = 0;
      nkin=0;
      nid=0;
      nacop=0;
      niso=0;
      nsel = 0;
      ncands =0; 
      nZ=0;
      neta=0;

      if(!plotHistograms_)  saveNTuple_ = false; // Just being careful...

      if(plotHistograms_) init_histograms();
     
}

void WMuNuSelector::init_histograms(){
     edm::Service<TFileService> fs;
     h1_["hNWCand"]                  =fs->make<TH1D>("NWCand","Nb. of WCandidates built",10,0.,10.);
     h1_["hNVvertex"]                =fs->make<TH1D>("NVvertex","Nb. of valid vertices",10,0.,10.);
     h1_["hNWCandPreSel"]            =fs->make<TH1D>("NWCandPreSel","Nb. of WCandidates passing pre-selection (ordered by pt)",10,0.,10.);
     h1_["hPtMu"]                    =fs->make<TH1D>("ptMu","Pt mu",100,0.,100.);
     h1_["hEtaMu"]                   =fs->make<TH1D>("etaMu","Eta mu",50,-2.5,2.5);
     h1_["hd0"]                      =fs->make<TH1D>("d0","Impact parameter",1000,-1.,1.);
     h1_["hMuonIDCuts"]              =fs->make<TH1D>("MuonIDQuality","Muon Passes all the VBTF Quality Criteria",2,-0.5,1.5);
     h1_["hMET"]                     =fs->make<TH1D>("MET","Missing Transverse Energy (GeV)", 200,0,200);
     h1_["hTMass"]                   =fs->make<TH1D>("TMass","Rec. Transverse Mass (GeV)",200,0,200);
     h1_["hAcop"]                    =fs->make<TH1D>("Acop","Mu-MET acoplanarity",50,0.,M_PI);
     h1_["hIsoTotN"]                 =fs->make<TH1D>("isoTotN","(Sum pT + Cal)/pT",1000,0.,10.);

     h1_["hCutFlowSummary"] = fs->make<TH1D>("CutFlowSummary", "Cut-flow Summary(number of events AFTER each cut)", 11, 0.5, 11.5);
     h1_["hCutFlowSummary"] ->GetXaxis()->SetBinLabel(1, "Total");
     h1_["hCutFlowSummary"] ->GetXaxis()->SetBinLabel(2, "Candidates");
     h1_["hCutFlowSummary"]->GetXaxis()->SetBinLabel(3, "HLT");
     h1_["hCutFlowSummary"]->GetXaxis()->SetBinLabel(4, "Z Rejection");
     h1_["hCutFlowSummary"]->GetXaxis()->SetBinLabel(5, "Jet Rejection.");
     h1_["hCutFlowSummary"]->GetXaxis()->SetBinLabel(6, "Muon ID Cuts");
     h1_["hCutFlowSummary"]->GetXaxis()->SetBinLabel(7, "Eta Cut");
     h1_["hCutFlowSummary"]->GetXaxis()->SetBinLabel(8, "Pt Cut");
     h1_["hCutFlowSummary"]->GetXaxis()->SetBinLabel(9, "Acop Cut");
     h1_["hCutFlowSummary"]->GetXaxis()->SetBinLabel(10, "Isolation Cut");
     h1_["hCutFlowSummary"]->GetXaxis()->SetBinLabel(11, "MET/MT Cut");


     if(saveNTuple_) { 
     hn_["hWntuple"] = fs->make<TNtuple>("Wntuple","Wmunu ntuple","quality:pt:eta:phi:d0:cal:sumpt:iso:acop:mt:met:phimet:sumEt:sig:charge");
     }   

}






void WMuNuSelector::endJob() {
      double all = nall;
      double epresel = npresel/all;
      double ecand = ncands/all;
      double etrig = ntrig/all;
      double ekin = nkin/all;
      double eid = nid/all;
      double eacop = nacop/all;
      double eiso = niso/all;
      double esel = nsel/all;

      LogVerbatim("") << "\n>>>>>> W SELECTION SUMMARY BEGIN >>>>>>>>>>>>>>>";
      LogVerbatim("") << "Total number of events analyzed                          : " << nall << " [events]";
      LogVerbatim("") << "Total number of events with WMuNuCandidates              : " << ncands << " [events]";
      LogVerbatim("") << "Total number of events triggered                         : " << ntrig << " [events]";
      LogVerbatim("") << "Total number of events pre-selected (jets/Z rejection)   : " << npresel << " [events]";
      LogVerbatim("") << "Total number of events after Muon ID cuts                : " << nid << " [events]";
      LogVerbatim("") << "Total number of events after kinematic cuts (pt,eta)     : " << nkin << " [events]";
      LogVerbatim("") << "Total number of events after Acop cut                    : " << nacop << " [events]";
      LogVerbatim("") << "Total number of events after iso cut                     : " << niso << " [events]";
      LogVerbatim("") << "Total number of events after MET/MT cut                  : " << nsel << " [events]";
      LogVerbatim("") << "Efficiencies CUMULATIVE:";
      LogVerbatim("") << "Candidate Building:                         " << "(" << setprecision(4) << ecand*100. <<" +/- "<< setprecision(2) << sqrt(ecand*(1-ecand)/all)*100. << ")%";
      LogVerbatim("") << "Trigger Efficiency:                         " << "(" << setprecision(4) << etrig*100. <<" +/- "<< setprecision(2) << sqrt(etrig*(1-etrig)/all)*100. << ")%";
      LogVerbatim("") << "Pre-Selection Efficiency:                   " << "(" << setprecision(4) << epresel*100. <<" +/- "<< setprecision(2) << sqrt(epresel*(1-epresel)/all)*100. << ")%";
      LogVerbatim("") << "MuonID Efficiency:                          " << "(" << setprecision(4) << eid*100. <<" +/- "<< setprecision(2) << sqrt(eid*(1-eid)/all)*100. << ")%";
      LogVerbatim("") << "Pt, Eta Selection Efficiency:               " << "(" << setprecision(4) << ekin*100. <<" +/- "<< setprecision(2) << sqrt(ekin*(1-ekin)/all)*100. << ")%";
      LogVerbatim("") << "Acop Efficiency:                            " << "(" << setprecision(4) << eacop*100. <<" +/- "<< setprecision(2) << sqrt(eacop*(1-eacop)/all)*100. << ")%";
      LogVerbatim("") << "Iso Efficiency:                             " << "(" << setprecision(4) << eiso*100. <<" +/- "<< setprecision(2) << sqrt(eiso*(1-eiso)/all)*100. << ")%";
      LogVerbatim("") << "Final Selection Efficiency (after MET/MT):  " << "(" << setprecision(4) << esel*100. <<" +/- "<< setprecision(2) << sqrt(esel*(1-esel)/nall)*100. << ")%";
      LogVerbatim("") << ">>>>>> W SELECTION SUMMARY END   >>>>>>>>>>>>>>>\n";
}

bool WMuNuSelector::filter (Event & ev, const EventSetup &) {
      nall++;
      if(plotHistograms_) h1_["hCutFlowSummary"]->Fill(1.);

      // Muon collection
      Handle<View<Muon> > muonCollection;
      if (!ev.getByLabel(muonTag_, muonCollection)) {
            LogError("") << ">>> Muon collection does not exist !!!";
            return 0;
      }
      unsigned int muonCollectionSize = muonCollection->size();


      // Trigger
      bool trigger_fired = false;
      if ( muonTrig_.size()==0){
            LogDebug("") << ">>> Careful, you are not requesting any trigger, event will be always fired";
      trigger_fired = true;
      }
      else {
            Handle<TriggerResults> triggerResults;
            if (!ev.getByLabel(trigTag_, triggerResults)) {
                  LogDebug("") << ">>> TRIGGER collection does not exist !!!";
            return 0;
            }
            const edm::TriggerNames & trigNames = ev.triggerNames(*triggerResults);
            LogDebug("")<<"Fired Triggers: ";

            for (unsigned int i=0; i<triggerResults->size(); i++)
            {
                  std::string trigName = trigNames.triggerName(i);
                  for (unsigned int j = 0; j < muonTrig_.size(); j++) {
                       if ( trigName == muonTrig_.at(j) && triggerResults->accept(i)) {
                              trigger_fired = true;
                       LogDebug("") <<"\t"<<trigName<<"   -->Trigger bit: "<<triggerResults->accept(i);

                  }
            }
            }
      } // Optionally you now have an OR of several triggers... Take care with the efficiencies in the analysis...



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
      LogDebug("") << "Total number of valid vertices / total vertices in the event: " << nvvertex<<" / "<<vertexCollectionSize;

      if(plotHistograms_) { h1_["hNVvertex"]->Fill(nvvertex); }

      if(NVertexMin_>nvvertex || nvvertex > NVertexMax_) { 
                  LogDebug("")<<"Wrong number of vertices"; return 0; 
      }






 
      // Loop to reject/control Z->mumu is done separately
      unsigned int nmuonsForZ1 = 0;
      unsigned int nmuonsForZ2 = 0;
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            double pt = mu.pt();
            if (pt>ptThrForZ1_) nmuonsForZ1++;
            if (pt>ptThrForZ2_) nmuonsForZ2++;
      }
      LogDebug("") << "> Z rejection: muons above " << ptThrForZ1_ << " [GeV]: " << nmuonsForZ1;
      LogDebug("") << "> Z rejection: muons above " << ptThrForZ2_ << " [GeV]: " << nmuonsForZ2;

      Handle<View<Jet> > jetCollection;
      if(jetTag_.label()==""){
      LogDebug("")<<">>> Careful, you are not requesting any jet collection";
      }else{
      // Jet collection
      if (!ev.getByLabel(jetTag_, jetCollection)) {
            LogError("") << ">>> JET collection does not exist !!!";
            LogError("") << " (if you dont want to check jets variables, edit your .py so that JetTag="", otherwise check your input file)";
            return 0;
      }
      }  

      // Beam spot
      Handle<reco::BeamSpot> beamSpotHandle;
      if (!ev.getByLabel(InputTag("offlineBeamSpot"), beamSpotHandle)) {
            LogDebug("") << ">>> No beam spot found !!!";
            return false;
      }



      // Get WMuNu candidates from file:

      Handle<reco::WMuNuCandidateCollection> WMuNuCollection;
      if (!ev.getByLabel(WMuNuCollectionTag_,WMuNuCollection) ) {
            LogDebug("") << ">>> WMuNu not found !!!";
            return false;
      }
 
      if(plotHistograms_){
             h1_["hNWCand"]->Fill(WMuNuCollection->size());
             // If the producer is set to only build 1 WMuNuCandidate per event, this plot is irrelevant..
      }

      ncands++;

       if(plotHistograms_) h1_["hCutFlowSummary"]->Fill(2.);
 
      if(WMuNuCollection->size() < 1) {LogDebug("")<<"No WMuNu Candidates in the Event!"; return 0;}
      if(WMuNuCollection->size() > 1) {LogDebug("")<<"This event contains more than one W Candidate";}  

      // W->mu nu selection criteria

      const WMuNuCandidate& WMuNu = WMuNuCollection->at(0);
      // By default there is only 1 WMuNu Candidate per event, the highest pt one
      // If you want to have all muons converted into WMuNuCandidates (for example, for a WZ analysis) modify the producer 
     
      const reco::Muon & mu = WMuNu.getMuon();
      const reco::MET  & met =WMuNu.getNeutrino();


      // Count Jets (in CSA07, a N<4 jets cut was used to reject ttbar. Not the default right now.)
      unsigned int jetCollectionSize = jetCollection->size();
      int njets = 0;
      for (unsigned int i=0; i<jetCollectionSize; i++) {
            const Jet& jet = jetCollection->at(i);
                  // This is in order to use PFJets as default jet collection 
                  // Muons are also PFJets, so in order to get a reasonable NJets plot you need to ensure that what you call "jet"
                  // is not actually your muon :-)
                  // IMPORTANT --> This module is intended for W inclusive analysis ONLY
                  // For V+Jets, there are specific analysis ongoing, go to ElectroWeakAnalysis/VJets      
                        double distance = sqrt( (mu.eta()-jet.eta())*(mu.eta()-jet.eta()) +(mu.phi()-jet.phi())*(mu.phi()-jet.phi()) );
                        // Matching in eta to see if the jet contains the muon      
                  if (distance<0.3) continue; // 0.3 is the isolation cone size around the muon
            if (jet.et()>eJetMin_) njets++;
      }
      LogDebug("") << ">>> Total number of jets: " << jetCollectionSize;
      LogDebug("") << ">>> Number of jets above " << eJetMin_ << " [GeV]: " << njets;


      // W->mu nu selection criteria
      LogDebug("") << "> WMuNu Candidate (charge="<<WMuNu.charge()<<"), with: ";


            if (!mu.isGlobalMuon()) return 0; 

            reco::TrackRef gm = mu.globalTrack();

            // Quality cuts
            double dxy = gm->dxy(beamSpotHandle->position());
            double normalizedChi2 = gm->normalizedChi2();
            int trackerHits = gm->hitPattern().numberOfValidTrackerHits();
            int pixelHits = gm->hitPattern().numberOfValidPixelHits();
            int muonHits = gm->hitPattern().numberOfValidMuonHits();
            int nMatches = mu.numberOfMatches();


            bool quality = true;

            if ( (dxy>dxyCut_) || (normalizedChi2>normalizedChi2Cut_) || (trackerHits<trackerHitsCut_) || (pixelHits<pixelHitsCut_) || (muonHits<muonHitsCut_) || (!mu.isTrackerMuon()) || (nMatches<nMatchesCut_) ) quality=false;


            LogDebug("") << "\t... dxy, normalizedChi2, muonhits, trackerHits, pixelHits, isTrackerMuon?, nMatches: " << dxy << " [cm], " << normalizedChi2 << ", " <<muonHits<<" , "<< trackerHits <<" , "<< pixelHits <<  ", " << mu.isTrackerMuon()<<", "<<nMatches;
            LogDebug("") << "\t... muon passes the quality cuts? "<<quality<<endl;


            // Kinematic variables:
            double pt = mu.pt();
            double eta = mu.eta();
            double phi = mu.phi();

            double acop = WMuNu.acop();

            double SumPt = mu.isolationR03().sumPt; double isovar=SumPt;
            double Cal   = mu.isolationR03().emEt + mu.isolationR03().hadEt; if(isCombinedIso_)isovar+=Cal;

            if (isRelativeIso_) isovar /= pt;
            bool iso = (isovar<=isoCut03_);
            LogDebug("") << "\t... isolation value" << isovar <<", isolated? " <<iso ;

            double met_et = met.pt();
            double sig = met.mEtSig();

            double massT = WMuNu.massT();
            double w_et = WMuNu.eT();

            // Preselection cuts:

            if (!trigger_fired) {LogDebug("")<<"Event did not fire the Trigger"; return 0;}
            ntrig++;
             if(plotHistograms_) h1_["hCutFlowSummary"]->Fill(3.);

            if (nmuonsForZ1>=1 && nmuonsForZ2>=2) {LogDebug("")<<"Z Candidate!!"; return 0;}
            nZ++;
             if(plotHistograms_) h1_["hCutFlowSummary"]->Fill(4.);

            if (njets>nJetMax_) {LogDebug("")<<"NJets > threshold";  return 0;}
            npresel++;
             if(plotHistograms_) h1_["hCutFlowSummary"]->Fill(5.);

            if(plotHistograms_){
             h1_["hNWCandPreSel"]->Fill(WMuNuCollection->size());
            }

     
            // Fill NTuple:
            if(saveNTuple_){
            hn_["hWntuple"]->Fill((double)quality,pt,eta,phi,dxy,Cal,SumPt,isovar,acop,massT,met_et,met.phi(),met.sumEt(),sig,WMuNu.charge());
            }

 
            // Select Ws by charge:

            if (selectByCharge_*WMuNu.charge()!=-1){ ncharge++;}
            else{return 0;}


            // Muon ID:


            if(plotHistograms_){ h1_["hMuonIDCuts"]->Fill(quality);}

            if(plotHistograms_){ h1_["hd0"]->Fill(dxy);}
            if (fabs(dxy)>dxyCut_) return 0;
            if (normalizedChi2>normalizedChi2Cut_) return 0;
            if (trackerHits<trackerHitsCut_) return 0;
            if (pixelHits<pixelHitsCut_) return 0;
            if (muonHits<muonHitsCut_) return 0;
            if (!mu.isTrackerMuon()) return 0;
            if (nMatches<nMatchesCut_) return 0;
            if(plotHistograms_)    h1_["hCutFlowSummary"]->Fill(6.);
            nid++;




            // Pt,eta cuts
            LogDebug("") << "\t... Muon pt, eta: " << pt << " [GeV], " << eta;

                    if(plotHistograms_){         
                              if (pt>=20.) h1_["hEtaMu"]->Fill(eta);
                        }
            if (fabs(eta)>etaCut_) return 0;
                   if(plotHistograms_) h1_["hCutFlowSummary"]->Fill(7.);
            neta++;

                  if(plotHistograms_){ h1_["hPtMu"]->Fill(pt);}
            if (pt<ptCut_) return 0;
            nkin++;
                   if(plotHistograms_) h1_["hCutFlowSummary"]->Fill(8.);

            // Acoplanarity cuts
            LogDebug("") << "\t... acoplanarity: " << acop;

            // Isolation cuts
                  if(plotHistograms_){
                  h1_["hIsoTotN"]->Fill( (SumPt+Cal) / pt );
                  }

            LogDebug("") << "\t... isolation value" << isovar <<", isolated? " <<iso ;
            LogDebug("") << "\t... Met  pt: "<<WMuNu.getNeutrino().pt()<<"[GeV]";
            LogDebug("") << "\t... W mass, W_et, W_px, W_py: " << massT << ", " << w_et << ", " << WMuNu.px() << ", " << WMuNu.py() << " [GeV]";

                 if(plotHistograms_){h1_["hAcop"]->Fill(acop);}
            if (acop>=acopCut_) return 0;

            nacop++;
                  if(plotHistograms_)  h1_["hCutFlowSummary"]->Fill(9.);


            if (!iso) return 0;
            niso++;

                  if(plotHistograms_){
                  h1_["hCutFlowSummary"]->Fill(10.);
                  h1_["hMET"]->Fill(met_et);
                  h1_["hTMass"]->Fill(massT);
                  }

            if (massT<=mtMin_ || massT>=mtMax_)  return 0;
            if (met_et<=metMin_ || met_et>=metMax_) return 0;

           LogDebug("") << ">>>> Event ACCEPTED";
            nsel++;
                  if(plotHistograms_) h1_["hCutFlowSummary"]->Fill(11.);


            // (To be continued ;-) )

            return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(WMuNuSelector);
