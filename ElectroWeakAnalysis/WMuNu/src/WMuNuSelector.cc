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
//    --> Be careful: if this Selector is used as a filter for further analysis you still have to make sure that        //
//                    the W Candidate you use for your modules is the first one in the collection!!                     //
//                                                                                                                      //
//    Optionally, plots selection variables sequentially after cuts,                                                    //
//    and 2D histograms for background determination.                                                                   //
//                                                                                                                      //
//    For basic plots before & after cuts (without Candidate formalism), use WMuNuValidator.cc                          //
//                                                                                                                      //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "AnalysisDataFormats/EWK/interface/WMuNuCandidate.h"
#include "TH1D.h"
#include "TH2D.h"

class WMuNuSelector : public edm::EDFilter {
public:
  WMuNuSelector (const edm::ParameterSet &);
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void beginJob() override;
  virtual void endJob() override;
  void init_histograms();
private:
  bool plotHistograms_;
  edm::EDGetTokenT<edm::TriggerResults> trigToken_;
  edm::EDGetTokenT<edm::View<reco::Muon> > muonToken_;
  edm::EDGetTokenT<edm::View<reco::Jet> > jetToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<reco::WMuNuCandidateCollection> WMuNuCollectionToken_;
  const std::string muonTrig_;
  double ptThrForZ1_;
  double ptThrForZ2_;
  double eJetMin_;
  int nJetMax_;
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
  bool isAlsoTrackerMuon_;

  int selectByCharge_;

  double nall;
  double ntrig, npresel;
  double nsel;
  double ncharge;
  double nkin, nid,nacop,niso,nmass;



  std::map<std::string,TH1D*> h1_;
  std::map<std::string,TH2D*> h2_;

};
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/METReco/interface/MET.h"

#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Common/interface/TriggerNames.h"


using namespace edm;
using namespace std;
using namespace reco;

WMuNuSelector::WMuNuSelector( const ParameterSet & cfg ) :
      // Fast selection (no histograms)
      plotHistograms_(cfg.getUntrackedParameter<bool> ("plotHistograms", true)),

      // Input collections
      trigToken_(consumes<TriggerResults>(cfg.getUntrackedParameter<edm::InputTag> ("TrigTag", edm::InputTag("TriggerResults::HLT")))),
      muonToken_(consumes<View<Muon> >(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons")))),
      jetToken_(consumes<View<Jet> >(cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("sisCone5CaloJets")))),
      beamSpotToken_(consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"))),
      WMuNuCollectionToken_(consumes<reco::WMuNuCandidateCollection>(cfg.getUntrackedParameter<edm::InputTag> ("WMuNuCollectionTag", edm::InputTag("WMuNus")))),


      // Preselection cuts
      muonTrig_(cfg.getUntrackedParameter<std::string>("MuonTrig", "HLT_Mu9")),
      ptThrForZ1_(cfg.getUntrackedParameter<double>("PtThrForZ1", 20.)),
      ptThrForZ2_(cfg.getUntrackedParameter<double>("PtThrForZ2", 10.)),
      eJetMin_(cfg.getUntrackedParameter<double>("EJetMin", 999999.)),
      nJetMax_(cfg.getUntrackedParameter<int>("NJetMax", 999999)),


      // Main cuts
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

   if(plotHistograms_){
     edm::Service<TFileService> fs;
     h1_["hNWCand_sel"]                  =fs->make<TH1D>("NWCand_sel","Nb. of WCandidates passing pre-selection (ordered by pt)",10,0.,10.);
     h1_["hPtMu_sel"]                    =fs->make<TH1D>("ptMu_sel","Pt mu",100,0.,100.);
     h1_["hEtaMu_sel"]                   =fs->make<TH1D>("etaMu_sel","Eta mu",50,-2.5,2.5);
     h1_["hd0_sel"]                      =fs->make<TH1D>("d0_sel","Impact parameter",1000,-1.,1.);
     h1_["hNHits_sel"]                   =fs->make<TH1D>("NumberOfValidHits_sel","Number of Hits in Silicon",100,0.,100.);
     h1_["hNormChi2_sel"]                =fs->make<TH1D>("NormChi2_sel","Chi2/ndof of global track",1000,0.,50.);
     h1_["hTracker_sel"]                 =fs->make<TH1D>("isTrackerMuon_sel","is Tracker Muon?",2,0.,2.);
     h1_["hMET_sel"]                     =fs->make<TH1D>("MET_sel","Missing Transverse Energy (GeV)", 300,0.,300.);
     h1_["hTMass_sel"]                   =fs->make<TH1D>("TMass_sel","Rec. Transverse Mass (GeV)",300,0.,300.);
     h1_["hAcop_sel"]                    =fs->make<TH1D>("Acop_sel","Mu-MET acoplanarity",50,0.,M_PI);
     h1_["hPtSum_sel"]                   =fs->make<TH1D>("ptSum_sel","Track Isolation, Sum pT (GeV)",200,0.,100.);
     h1_["hPtSumN_sel"]                  =fs->make<TH1D>("ptSumN_sel","Track Isolation, Sum pT/pT",1000,0.,10.);
     h1_["hCal_sel"]                     =fs->make<TH1D>("Cal_sel","Calorimetric isolation, HCAL+ECAL (GeV)",200,0.,100.);
     h1_["hIsoTotN_sel"]                 =fs->make<TH1D>("isoTotN_sel","(Sum pT + Cal)/pT",1000,0.,10.);
     h1_["hIsoTot_sel"]                  =fs->make<TH1D>("isoTot_sel","(Sum pT + Cal)",200,0.,100.);
     h2_["hTMass_PtSum_inclusive"]       =fs->make<TH2D>("TMass_PtSum_inclusive","Rec. Transverse Mass (GeV) vs Sum pT (GeV)",200,0.,100.,300,0.,300.);
     h2_["hTMass_PtSumNorm_inclusive"]   =fs->make<TH2D>("TMass_PtSumNorm_inclusive","Rec. Transverse Mass (GeV) vs Sum Pt / Pt", 1000,0,10,300,0,300);
     h2_["hTMass_TotIsoNorm_inclusive"]=fs->make<TH2D>("TMass_TotIsoNorm_inclusive","Rec. Transverse Mass (GeV) vs (Sum Pt + Cal)/Pt", 10000,0,10,200,0,200);
     h2_["hMET_PtSum_inclusive"]         =fs->make<TH2D>("MET_PtSum_inclusive","Missing Transverse Energy (GeV) vs Sum Pt (GeV)",200,0.,100.,300,0.,300.);
     h2_["hMET_PtSumNorm_inclusive"]     =fs->make<TH2D>("MET_PtSumNorm_inclusive","Missing Transverse Energy (GeV) vs Sum Pt/Pt",1000,0,10,300,0,300);
     h2_["hMET_TotIsoNorm_inclusive"]=fs->make<TH2D>("MET_TotIsoNorm_inclusive","Missing Transverse Energy (GeV) vs (SumPt + Cal)/Pt",10000,0,10,200,0,200);
   }
}


void WMuNuSelector::endJob() {
      double all = nall;
      double epresel = npresel/all;
      double etrig = ntrig/all;
      double ekin = nkin/all;
      double eid = nid/all;
      double eacop = nacop/all;
      double eiso = niso/all;
      double esel = nsel/all;

      LogVerbatim("") << "\n>>>>>> W SELECTION SUMMARY BEGIN >>>>>>>>>>>>>>>";
      LogVerbatim("") << "Total number of events analyzed: " << nall << " [events]";
      LogVerbatim("") << "Total number of events triggered: " << ntrig << " [events]";
      LogVerbatim("") << "Total number of events pre-selected: " << npresel << " [events]";
      LogVerbatim("") << "Total number of events after kinematic cuts: " << nkin << " [events]";
      LogVerbatim("") << "Total number of events after Muon ID cuts: " << nid << " [events]";
      LogVerbatim("") << "Total number of events after Acop cut: " << nacop << " [events]";
      LogVerbatim("") << "Total number of events after iso cut: " << niso << " [events]";
      LogVerbatim("") << "Total number of events selected: " << nsel << " [events]";
      LogVerbatim("") << "Efficiencies:";
      LogVerbatim("") << "Trigger Efficiency:                   " << "(" << setprecision(4) << etrig*100. <<" +/- "<< setprecision(2) << sqrt(etrig*(1-etrig)/all)*100. << ")%";
      LogVerbatim("") << "Pre-Selection Efficiency:             " << "(" << setprecision(4) << epresel*100. <<" +/- "<< setprecision(2) << sqrt(epresel*(1-epresel)/all)*100. << ")%";
      LogVerbatim("") << "Pt, Eta Selection Efficiency:         " << "(" << setprecision(4) << ekin*100. <<" +/- "<< setprecision(2) << sqrt(ekin*(1-ekin)/all)*100. << ")%";
      LogVerbatim("") << "MuonID Efficiency:                    " << "(" << setprecision(4) << eid*100. <<" +/- "<< setprecision(2) << sqrt(eid*(1-eid)/all)*100. << ")%";
      LogVerbatim("") << "Acop Efficiency:                      " << "(" << setprecision(4) << eacop*100. <<" +/- "<< setprecision(2) << sqrt(eacop*(1-eacop)/all)*100. << ")%";
      LogVerbatim("") << "Iso Efficiency:                       " << "(" << setprecision(4) << eiso*100. <<" +/- "<< setprecision(2) << sqrt(eiso*(1-eiso)/all)*100. << ")%";
      LogVerbatim("") << "Selection Efficiency:             " << "(" << setprecision(4) << esel*100. <<" +/- "<< setprecision(2) << sqrt(esel*(1-esel)/nall)*100. << ")%";

     if ( fabs(selectByCharge_)==1 ){
      esel = nsel/ncharge;
      LogVerbatim("") << "\n>>>>>> W+(-) SELECTION >>>>>>>>>>>>>>>";
      LogVerbatim("") << "Total number of W+(-) events pre-selected: " << ncharge << " [events]";
      LogVerbatim("") << "Total number of events selected: " << nsel << " [events]";
      LogVerbatim("") << "Selection Efficiency only W+(-): " << "(" << setprecision(4) << esel*100. <<" +/- "<< setprecision(2) << sqrt(esel*(1-esel)/ncharge)*100. << ")%";
     }
      LogVerbatim("") << ">>>>>> W SELECTION SUMMARY END   >>>>>>>>>>>>>>>\n";
}

bool WMuNuSelector::filter (Event & ev, const EventSetup &) {
      nall++;

      // Repeat Pre-Selection Cuts just in case...
      // Muon collection
      Handle<View<Muon> > muonCollection;
      if (!ev.getByToken(muonToken_, muonCollection)) {
            LogError("") << ">>> Muon collection does not exist !!!";
            return 0;
      }
      unsigned int muonCollectionSize = muonCollection->size();

      // Trigger
      Handle<TriggerResults> triggerResults;
      if (!ev.getByToken(trigToken_, triggerResults)) {
            LogError("") << ">>> TRIGGER collection does not exist !!!";
            return 0;
      }
      const edm::TriggerNames & triggerNames = ev.triggerNames(*triggerResults);
      bool trigger_fired = false;
      int itrig1 = triggerNames.triggerIndex(muonTrig_);
      if (triggerResults->accept(itrig1)) trigger_fired = true;
      LogTrace("") << ">>> Trigger bit: " << trigger_fired << " (" << muonTrig_ << ")";

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
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ1_ << " [GeV]: " << nmuonsForZ1;
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ2_ << " [GeV]: " << nmuonsForZ2;

      // Jet collection
      Handle<View<Jet> > jetCollection;
      if (!ev.getByToken(jetToken_, jetCollection)) {
            LogError("") << ">>> JET collection does not exist !!!";
            return 0;
      }
      unsigned int jetCollectionSize = jetCollection->size();
      int njets = 0;
      for (unsigned int i=0; i<jetCollectionSize; i++) {
            const Jet& jet = jetCollection->at(i);
            if (jet.et()>eJetMin_) njets++;
      }
      LogTrace("") << ">>> Total number of jets: " << jetCollectionSize;
      LogTrace("") << ">>> Number of jets above " << eJetMin_ << " [GeV]: " << njets;


      // Beam spot
      Handle<reco::BeamSpot> beamSpotHandle;
      if (!ev.getByToken(beamSpotToken_, beamSpotHandle)) {
            LogTrace("") << ">>> No beam spot found !!!";
            return false;
      }



      // Get WMuNu candidates from file:

      Handle<reco::WMuNuCandidateCollection> WMuNuCollection;
      if (!ev.getByToken(WMuNuCollectionToken_,WMuNuCollection) ) {
            LogTrace("") << ">>> WMuNu not found !!!";
            return false;
      }

      if(WMuNuCollection->size() < 1) {LogTrace("")<<"No WMuNu Candidates in the Event!"; return 0;}
      if(WMuNuCollection->size() > 1) {LogTrace("")<<"This event contains more than one W Candidate";}

      // W->mu nu selection criteria

      LogTrace("") << "> WMuNu Candidate with: ";
      const WMuNuCandidate& WMuNu = WMuNuCollection->at(0);
      // WMuNuCandidates are ordered by Pt!
      // The Inclusive Selection WMuNu Candidate is the first one

      const reco::Muon & mu = WMuNu.getMuon();
      const reco::MET  & met =WMuNu.getNeutrino();
            if(plotHistograms_){
            h1_["hNWCand_sel"]->Fill(WMuNuCollection->size());
            }


      // Preselection cuts:

      if (!trigger_fired) {LogTrace("")<<"Event did not fire the Trigger"; return 0;}
      ntrig++;

      if (nmuonsForZ1>=1 && nmuonsForZ2>=2) {LogTrace("")<<"Z Candidate!!"; return 0;}
      if (njets>nJetMax_) {LogTrace("")<<"NJets > threshold";  return 0;}

      npresel++;

      // Select Ws by charge:

      if (selectByCharge_*WMuNu.charge()==-1){ ncharge++;}


      // W->mu nu selection criteria

            if (!mu.isGlobalMuon()) return 0;

            reco::TrackRef gm = mu.globalTrack();
            //reco::TrackRef tk = mu.innerTrack();

            // Pt,eta cuts
            double pt = mu.pt();
            double eta = mu.eta();
            LogTrace("") << "\t... Muon pt, eta: " << pt << " [GeV], " << eta;
                  if(plotHistograms_){ h1_["hPtMu_sel"]->Fill(pt);}
            if (pt<ptCut_) return 0;
                  if(plotHistograms_){ h1_["hEtaMu_sel"]->Fill(eta);}
            if (fabs(eta)>etaCut_) return 0;

            nkin++;


            // d0, chi2, nhits quality cuts
            double dxy = gm->dxy(beamSpotHandle->position());
            double normalizedChi2 = gm->normalizedChi2(); LogTrace("")<<"Im here"<<endl;
            double trackerHits = gm->hitPattern().numberOfValidTrackerHits();
            LogTrace("") << "\t... Muon dxy, normalizedChi2, trackerHits, isTrackerMuon?: " << dxy << " [cm], "<<normalizedChi2 << ", "<<trackerHits << ", " << mu.isTrackerMuon();

                  if(plotHistograms_){ h1_["hd0_sel"]->Fill(dxy);}
            if (!muon::isGoodMuon(mu,muon::GlobalMuonPromptTight) ) return 0;
                  if(plotHistograms_){ h1_["hNormChi2_sel"]->Fill(normalizedChi2);}
            if (normalizedChi2>normalizedChi2Cut_) return 0;
                  if(plotHistograms_){ h1_["hNHits_sel"]->Fill(trackerHits);}
            if (trackerHits<trackerHitsCut_) return 0;
                  if(plotHistograms_){ h1_["hTracker_sel"]->Fill(mu.isTrackerMuon());}
            if (!mu.isTrackerMuon()) return 0;

            nid++;

            // Acoplanarity cuts
            double acop = WMuNu.acop();
            LogTrace("") << "\t... acoplanarity: " << acop;

            // Isolation cuts
            double SumPt = mu.isolationR03().sumPt; double isovar=SumPt;
            double Cal   = mu.isolationR03().emEt + mu.isolationR03().hadEt; if(isCombinedIso_)isovar+=Cal;
                  if(plotHistograms_){
                  h1_["hPtSum_sel"]->Fill(SumPt);
                  h1_["hPtSumN_sel"]->Fill(SumPt/pt);
                  h1_["hCal_sel"]->Fill(Cal);
                  h1_["hIsoTot_sel"]->Fill( (SumPt+Cal));
                  h1_["hIsoTotN_sel"]->Fill( (SumPt+Cal) / pt );
                  }

            if (isRelativeIso_) isovar /= pt;
            bool iso = (isovar<=isoCut03_);
            LogTrace("") << "\t... isolation value" << isovar <<", isolated? " <<iso ;

            double met_et = met.pt();
            LogTrace("") << "\t... Met  pt: "<<WMuNu.getNeutrino().pt()<<"[GeV]";


            double massT = WMuNu.massT();
            double w_et = WMuNu.eT();

            LogTrace("") << "\t... W mass, W_et, W_px, W_py: " << massT << ", " << w_et << ", " << WMuNu.px() << ", " << WMuNu.py() << " [GeV]";


            // Plot 2D Histograms before final cuts
                  if(plotHistograms_ && acop<acopCut_){
                  h2_["hTMass_PtSum_inclusive"]->Fill(SumPt,massT);
                  h2_["hTMass_PtSumNorm_inclusive"]->Fill(SumPt/pt,massT);
                  h2_["hTMass_TotIsoNorm_inclusive"]->Fill((SumPt+Cal)/pt,massT);
                  h2_["hMET_PtSum_inclusive"]->Fill(SumPt,met_et);
                  h2_["hMET_PtSumNorm_inclusive"]->Fill(SumPt/pt,met_et);
                  h2_["hMET_TotIsoNorm_inclusive"]->Fill((SumPt+Cal)/pt,met_et);
                  }

            if (!iso) return 0;

            niso++;

             if(plotHistograms_){ h1_["hAcop_sel"]->Fill(acop);}
            if (acop>=acopCut_) return 0;

           nacop++;

            if(plotHistograms_){
                  h1_["hMET_sel"]->Fill(met_et);
                  h1_["hTMass_sel"]->Fill(massT);
            }


            if (massT<=mtMin_ || massT>=mtMax_)  return 0;
            if (met_et<=metMin_ || met_et>=metMax_) return 0;

           LogTrace("") << ">>>> Event ACCEPTED";
            nsel++;


            // (To be continued ;-) )

            return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(WMuNuSelector);
