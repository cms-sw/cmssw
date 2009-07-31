//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                      //
//                                    WMuNuSelector based on WMuNuCandidates                                            //
//                                                                                                                      //    
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                      //      
//    Filter of WMuNuCandidates for Analysis                                                                            //
//    --> From a WMuNuCandidate collection with event cuts (trigger, Z rejection, ttbar rejection) already in place     //
//    --> Ws are pre-selected from the highest pt muon in the event (once Zs have been rejected)                        //
//                                                                                                                      //
//    Optionally, plots selection variables sequentially after cuts,                                                       //
//    and 2D histograms for background determination.                                                                   //
//                                                                                                                      //
//    For basic plots before & after cuts (without Candidate formalis), use WMuNuValidator.cc                           //
//                                                                                                                      //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "TH1D.h"
#include "TH2D.h"

class WMuNuSelector : public edm::EDFilter {
public:
  WMuNuSelector (const edm::ParameterSet &);
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();
  void init_histograms();
private:
  bool plotHistograms_;
  edm::InputTag muonTag_;
  edm::InputTag metTag_;
  bool metIncludesMuons_;
  edm::InputTag jetTag_;
  edm::InputTag WMuNuCollectionTag_;
  bool useTrackerPt_;
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
  double nsel;
  double ncharge; 

  std::map<std::string,TH1D*> h1_;
  std::map<std::string,TH2D*> h2_;

};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/Common/interface/View.h"

#include "ElectroWeakAnalysis/WMuNu/interface/WMuNuCandidate.h"

  
using namespace edm;
using namespace std;
using namespace reco;

WMuNuSelector::WMuNuSelector( const ParameterSet & cfg ) :
      // Fast selection (no histograms)
      plotHistograms_(cfg.getUntrackedParameter<bool> ("PlotHistograms", false)),

      // Input collections
      muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met"))),
      metIncludesMuons_(cfg.getUntrackedParameter<bool> ("METIncludesMuons", false)),
      jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("sisCone5CaloJets"))),
      WMuNuCollectionTag_(cfg.getUntrackedParameter<edm::InputTag> ("WMuNuCollectionTag", edm::InputTag("WMuNus"))),

      // Main cuts 
      useTrackerPt_(cfg.getUntrackedParameter<bool>("UseTrackerPt", true)),
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

void WMuNuSelector::beginJob(const EventSetup &) {
      nall = 0;
      nsel = 0;
      ncharge = 0;

   if(!plotHistograms_){
     edm::Service<TFileService> fs;
     h1_["hNWCand_sel"]                  =fs->make<TH1D>("NWCand_sel","Nb. of WCandidates passing pre-selection",10,0.,10.);
     h1_["hPtMu_sel"]                    =fs->make<TH1D>("ptMu_sel","Pt mu",100,0.,100.);
     h1_["hEtaMu_sel"]                   =fs->make<TH1D>("etaMu_sel","Eta mu",50,-2.5,2.5);
     h1_["hd0_sel"]                      =fs->make<TH1D>("d0_sel","Impact parameter",1000,-1.,1.);
     h1_["hNumberOfValidHits_sel"]       =fs->make<TH1D>("NumberOfValidHits_sel","Number of Hits in Silicon",100,0.,100.);
     h1_["hNormChi2_sel"]                =fs->make<TH1D>("NormChi2_sel","Chi2/ndof of global track",1000,0.,50.);
     h1_["hTracker_sel"]                 =fs->make<TH1D>("isTrackerMuon_sel","is Tracker Muon?",2,0.,2.); 
     h1_["hMET_sel"]                     =fs->make<TH1D>("MET_sel","Missing Transverse Energy (GeV)", 300,0.,300.);
     h1_["hTMass_sel"]                   =fs->make<TH1D>("TMass_sel","Rec. Transverse Mass (GeV)",300,0.,300.);
     h1_["hAcop_sel"]                    =fs->make<TH1D>("Acop_sel","Mu-MET acoplanarity",50,0.,M_PI);
     h1_["hPtSum_sel"]                   =fs->make<TH1D>("ptSum_sel","Track Isolation, Sum pT (GeV)",200,0.,100.);
     h1_["hPtSumN_sel"]                  =fs->make<TH1D>("ptSumN_sel","Track Isolation, Sum pT/pT",1000,0.,10.);
     h1_["hCal_sel"]                     =fs->make<TH1D>("Cal_sel","Calorimetric isolation, HCAL+ECAL (GeV)",200,0.,100.);
     h1_["hIsoTotN_sel"]                 =fs->make<TH1D>("isoTotN_sel","(Sum pT + Cal)/pT",1000,0.,10.);
     h2_["hTMass_PtSum_inclusive"]       =fs->make<TH2D>("TMass_PtSum_inclusive","Rec. Transverse Mass (GeV) vs Sum pT (GeV)",200,0.,100.,300,0.,300.);
     h2_["hTMass_PtSumNorm_inclusive"]   =fs->make<TH2D>("TMass_PtSumNorm_inclusive","Rec. Transverse Mass (GeV) vs Sum Pt / Pt", 1000,0,10,300,0,300);
     h2_["hTMass_TotIsoNorm_inclusive"]=fs->make<TH2D>("TMass_TotIsoNorm_inclusive","Rec. Transverse Mass (GeV) vs (Sum Pt + Cal)/Pt", 10000,0,10,200,0,200);
     h2_["hMET_PtSum_inclusive"]         =fs->make<TH2D>("MET_PtSum_inclusive","Missing Transverse Energy (GeV) vs Sum Pt (GeV)",200,0.,100.,300,0.,300.);
     h2_["hMET_PtSumNorm_inclusive"]     =fs->make<TH2D>("MET_PtSumNorm_inclusive","Missing Transverse Energy (GeV) vs Sum Pt/Pt",1000,0,10,300,0,300);
     h2_["hMET_TotIsoNorm_inclusive"]=fs->make<TH2D>("MET_TotIsoNorm_inclusive","Missing Transverse Energy (GeV) vs (SumPt + Cal)/Pt",10000,0,10,200,0,200);
   }
}


void WMuNuSelector::endJob() {
      double esel = nsel/nall;
      LogVerbatim("") << "\n>>>>>> W SELECTION SUMMARY BEGIN >>>>>>>>>>>>>>>";
      LogVerbatim("") << "Total numer of events analyzed: " << nall << " [events]";
      LogVerbatim("") << "Total numer of events selected: " << nsel << " [events]";
      LogVerbatim("") << "Overall efficiency:             " << "(" << setprecision(4) << esel*100. <<" +/- "<< setprecision(2) << sqrt(esel*(1-esel)/nall)*100. << ")%";

     if ( fabs(selectByCharge_)==1 ){
      esel = nsel/ncharge;
      LogVerbatim("") << "\n>>>>>> W+(-) SELECTION >>>>>>>>>>>>>>>";
      LogVerbatim("") << "Total numer of W+(-) events analyzed: " << ncharge << " [events]";
      LogVerbatim("") << "Total numer of events selected: " << nsel << " [events]";
      LogVerbatim("") << "Overall efficiency selecting only W+(-): " << "(" << setprecision(4) << esel*100. <<" +/- "<< setprecision(2) << sqrt(esel*(1-esel)/ncharge)*100. << ")%";    
     }
      LogVerbatim("") << ">>>>>> W SELECTION SUMMARY END   >>>>>>>>>>>>>>>\n";
}

bool WMuNuSelector::filter (Event & ev, const EventSetup &) {

      // Beam spot
      Handle<reco::BeamSpot> beamSpotHandle;
      if (!ev.getByLabel(InputTag("offlineBeamSpot"), beamSpotHandle)) {
            LogTrace("") << ">>> No beam spot found !!!";
            return false;
      }

      // Get WMuNu candidates from file:

      Handle<reco::WMuNuCandidateCollection> WMuNuCollection;
      if (!ev.getByLabel(WMuNuCollectionTag_,WMuNuCollection) ) {
            LogTrace("") << ">>> WMuNu not found !!!";
            return false;
      }
  

      // W->mu nu selection criteria

      LogTrace("") << "> WMuNu Candidate with: ";
      const WMuNuCandidate& WMuNu = WMuNuCollection->at(0);
      const reco::Muon & mu = WMuNu.getMuon();
      const reco::MET  & met =WMuNu.getNeutrino();
            if(plotHistograms_){
             //h1_["hNWCand_sel"]->Fill(WMuNuCollection.size());  
            }
      nall++;


      // Select Ws by charge:  
  
      if (selectByCharge_*WMuNu.charge()==-1)
      ncharge++;


      // W->mu nu selection criteria

            if (!mu.isGlobalMuon()) return 0; 

            reco::TrackRef gm = mu.globalTrack();
            reco::TrackRef tk = mu.innerTrack();

            // Pt,eta cuts
            double pt = mu.pt();
            if (useTrackerPt_) pt = tk->pt();
            double eta = mu.eta();
            LogTrace("") << "\t... Muon pt, eta: " << pt << " [GeV], " << eta;
                  if(!plotHistograms_){ h1_["hPtMu_sel"]->Fill(pt);}
            if (pt<ptCut_) return 0;

                  if(!plotHistograms_){ h1_["hEtaMu_sel"]->Fill(eta);}
            if (fabs(eta)>etaCut_) return 0;

            // d0, chi2, nhits quality cuts
            double dxy = tk->dxy(beamSpotHandle->position());
            double normalizedChi2 = gm->normalizedChi2();
            double trackerHits = tk->numberOfValidHits();
            LogTrace("") << "\t... Muon dxy, normalizedChi2, trackerHits, isTrackerMuon?: " << dxy << " [cm], " << normalizedChi2 << ", " << trackerHits << ", " << mu.isTrackerMuon();
                  if(!plotHistograms_){ h1_["hd0_sel"]->Fill(dxy);}
            if (fabs(dxy)>dxyCut_) return 0;
                  if(!plotHistograms_){ h1_["hNormChi2_sel"]->Fill(normalizedChi2);}
            if (normalizedChi2>normalizedChi2Cut_) return 0;
                  if(!plotHistograms_){ h1_["hNHits_sel"]->Fill(trackerHits);}
            if (trackerHits<trackerHitsCut_) return 0;
                  if(!plotHistograms_){ h1_["hTracker_sel"]->Fill(mu.isTrackerMuon());}
            if (!mu.isTrackerMuon()) return 0;

            // Acoplanarity cuts
            double acop = WMuNu.acop();
            LogTrace("") << "\t... acoplanarity: " << acop;
                  if(!plotHistograms_){ h1_["hAcop_sel"]->Fill(acop);}
            if (acop>acopCut_) return 0;



            // Isolation cuts
            double SumPt = mu.isolationR03().sumPt; double isovar=SumPt;
            double Cal   = mu.isolationR03().emEt + mu.isolationR03().hadEt; if(isCombinedIso_)isovar+=Cal;            
                  if(!plotHistograms_){
                  h1_["hPtSum_sel"]->Fill(SumPt);
                  h1_["hPtSumN_sel"]->Fill(SumPt/pt);
                  h1_["hCal_sel"]->Fill(Cal);
                  h1_["hIsoTot_sel"]->Fill( (SumPt+Cal));
                  h1_["hIsoTotN_sel"]->Fill( (SumPt+Cal) / pt );
                  }

            if (isRelativeIso_) isovar /= pt;
            bool iso = (isovar<isoCut03_);
            LogTrace("") << "\t... isolation value" << isovar <<", isolated? " <<iso ;

            double met_et = met.pt();
            LogTrace("") << "\t... Met  pt: "<<WMuNu.getNeutrino().pt()<<"[GeV]";


            double massT = WMuNu.massT(useTrackerPt_);
            double w_et = WMuNu.eT(useTrackerPt_);

            LogTrace("") << "\t... W mass, W_et, W_px, W_py: " << massT << ", " << w_et << ", " << WMuNu.px() << ", " << WMuNu.py() << " [GeV]";

            // Plot 2D Histograms before final cuts 
                  if(!plotHistograms_){
                  h2_["hTMass_PtSum_inclusive"]->Fill(SumPt,massT);   
                  h2_["hTMass_PtSumNorm_inclusive"]->Fill(SumPt/pt,massT);
                  h2_["hTMass_TotIsoNorm_inclusive"]->Fill((SumPt+Cal)/pt,massT);
                  h2_["hMET_PtSum_inclusive"]->Fill(SumPt,met_et);   
                  h2_["hMET_PtSumNorm_inclusive"]->Fill(SumPt/pt,met_et);
                  h2_["hMET_TotIsoNorm_inclusive"]->Fill((SumPt+Cal)/pt,met_et);
                  }     

            if (!iso) return 0;
            
            if(!plotHistograms_){
                  h1_["hMET_sel"]->Fill(met_et);
                  h1_["hTMass_sel"]->Fill(massT);
            }


            if (massT<mtMin_ && massT>mtMax_)  return 0;
            if (met_et<metMin_ && met_et>metMax_) return 0;

            LogTrace("") << ">>>> Event ACCEPTED";
            nsel++;


            // (To be continued)

            return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(WMuNuSelector);
