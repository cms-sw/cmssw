/** \class WMuNuAnalyzer
 *  Simple analyzer to make some W->MuNu plots
 *
 *  \author M.I. Josa
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class TFile;
class TH1F;
class TH2F;


class WMuNuAnalyzer : public edm::EDAnalyzer {
public:
  WMuNuAnalyzer(const edm::ParameterSet& pset);
  virtual ~WMuNuAnalyzer();
  virtual void beginJob(const edm::EventSetup& eventSetup);
  virtual void endJob();
  virtual void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);
private:
  edm::InputTag muonTag_;
  edm::InputTag metTag_;
  edm::InputTag jetTag_;
  bool useOnlyGlobalMuons_;
  double ptCut_;
  double etaCut_;
  bool isRelativeIso_;
  double isoCut03_;
  double massTMin_;
  double massTMax_;
  double ptThrForZCount_;
  double acopCut_;
  double eJetMin_;
  int nJetMax_;

// Histograms
  TH1F *hNMu;
  TH1F *hPtMu;
  TH1F *hEtaMu;
  TH1F *hMET;
  TH1F *hTMass;
  TH1F *hAcop;
  TH1F *hNjets;
  TH1F *hPtSum;
  TH1F *hPtSumN;
  TH2F *hTMass_PtSum;

  TH1F *hNMu_sel;
  TH1F *hPtMu_sel;
  TH1F *hEtaMu_sel;
  TH1F *hMET_sel;
  TH1F *hTMass_sel;
  TH1F *hAcop_sel;
  TH1F *hNjets_sel;
  TH1F *hPtSum_sel;
  TH1F *hPtSumN_sel;
  TH2F *hTMass_PtSum_sel;

  // Root output file
  std::string theRootFileName;
  TFile* theRootFile;

  unsigned int numberOfEvents;
  unsigned int numberOfMuons;
};

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/GeometryVector/interface/Phi.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

#include <map>
#include <vector>

using namespace std;
using namespace edm;
using namespace reco;

/// Constructor
WMuNuAnalyzer::WMuNuAnalyzer(const ParameterSet& pset) :
      muonTag_(pset.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_(pset.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met"))),
      jetTag_(pset.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("iterativeCone5CaloJets"))),
      useOnlyGlobalMuons_(pset.getUntrackedParameter<bool>("UseOnlyGlobalMuons", true)),
      ptCut_(pset.getUntrackedParameter<double>("PtCut", 25.)),
      etaCut_(pset.getUntrackedParameter<double>("EtaCut", 2.1)),
      isRelativeIso_(pset.getUntrackedParameter<bool>("IsRelativeIso", true)),
      isoCut03_(pset.getUntrackedParameter<double>("IsoCut03", 0.1)),
      massTMin_(pset.getUntrackedParameter<double>("MassTMin", 50.)),
      massTMax_(pset.getUntrackedParameter<double>("MassTMax", 200.)),
      ptThrForZCount_(pset.getUntrackedParameter<double>("PtThrForZCount", 20.)),
      acopCut_(pset.getUntrackedParameter<double>("AcopCut", 999999.)),
      eJetMin_(pset.getUntrackedParameter<double>("EJetMin", 999999.)),
      nJetMax_(pset.getUntrackedParameter<int>("NJetMax", 999999)),
      theRootFileName(pset.getUntrackedParameter<string>("rootFileName")) 

{
  LogDebug("WMuNuAnalyzer")<<" WMuNuAnalyzer constructor called";
}

/// Destructor
WMuNuAnalyzer::~WMuNuAnalyzer(){
}

void WMuNuAnalyzer::beginJob(const EventSetup& eventSetup){
  // Create output files

  theRootFile = TFile::Open(theRootFileName.c_str(), "RECREATE");
  theRootFile->cd();

  numberOfEvents = 0;
  numberOfMuons = 0;

  hNMu    = new TH1F("NMu","Nb. muons in the event",10,0.,10.);
  hPtMu   = new TH1F("ptMu","Pt mu",100,0.,100.);
  hEtaMu  = new TH1F("etaMu","Eta mu",50,-2.5,2.5);
  hMET    = new TH1F("MET","Missing Transverse Energy (GeV)", 100,0.,200.);
  hTMass  = new TH1F("TMass","Rec. Transverse Mass (GeV)",150,0.,300.);
  hAcop   = new TH1F("Acop","Mu-MET acoplanarity",50,0.,M_PI);
  hNjets  = new TH1F("Njets","njets",25,0.,25.);
  hPtSum  = new TH1F("ptSum","Sum pT (GeV)",100,0.,50.);
  hPtSumN = new TH1F("ptSumN","Sum pT/pT",100,0.,50.);
  hTMass_PtSum = new TH2F("TMass_ptSum","Rec. Transverse Mass (GeV) vs Sum pT (GeV)",100,0.,50.,150,0.,300.);

  hNMu_sel    = new TH1F("NMu_sel","Nb. selected muons",10,0.,10.);
  hPtMu_sel   = new TH1F("ptMu_sel","Pt mu",100,0.,100.);
  hEtaMu_sel  = new TH1F("etaMu_sel","Eta mu",50,-2.5,2.5);
  hMET_sel    = new TH1F("MET_sel","Missing Transverse Energy (GeV)", 100,0.,200.);
  hTMass_sel  = new TH1F("TMass_sel","Rec. Transverse Mass (GeV)",150,0.,300.);
  hAcop_sel   = new TH1F("Acop_sel","Mu-MET acoplanarity",50,0.,M_PI);
  hNjets_sel  = new TH1F("Njets_sel","njets",25,0.,25.);
  hPtSum_sel  = new TH1F("ptSum_sel","Sum pT (GeV)",100,0.,50.);
  hPtSumN_sel = new TH1F("ptSumN_sel","Sum pT/pT ",100,0.,2.5);
  hTMass_PtSum_sel = new TH2F("TMass_ptSum_sel","Rec. Transverse Mass (GeV) vs Sum pT (GeV)",100,0.,50.,150,0.,300.);

}

void WMuNuAnalyzer::endJob(){
  LogVerbatim("") << "WMuNuAnalyzer>>> FINAL PRINTOUTS -> BEGIN";
  LogVerbatim("") << "WMuNuAnalyzer>>> Number of analyzed events= " << numberOfEvents;
  LogVerbatim("") << "WMuNuAnalyzer>>> Number of analyzed muons= " << numberOfMuons;

  // Write the histos to file
  theRootFile->cd();

  hNMu->Write();
  hPtMu->Write();   
  hEtaMu->Write();  
  hMET->Write();
  hTMass->Write(); 
  hAcop->Write(); 
  hNjets->Write(); 
  hPtSum->Write(); 
  hPtSumN->Write(); 
  hTMass_PtSum->Write(); 

  hNMu_sel->Write();
  hPtMu_sel->Write();   
  hEtaMu_sel->Write();  
  hMET_sel->Write();
  hTMass_sel->Write(); 
  hAcop_sel->Write(); 
  hNjets_sel->Write(); 
  hPtSum_sel->Write(); 
  hPtSumN_sel->Write(); 
  hTMass_PtSum_sel->Write(); 

  theRootFile->Close();

  LogVerbatim("") << "WMuNuAnalyzer>>> FINAL PRINTOUTS -> END";
}
 

void WMuNuAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  
   numberOfEvents++;

   double pt_sel[5];
   double eta_sel[5];
   double acop_sel[5];
   double massT_sel[5];
   double iso_sel[5];
   double isoN_sel[5];

   bool event_sel = true;
  
   double met_px = 0.;
   double met_py = 0.;

  // Get the Muon Track collection from the event
   Handle<reco::MuonCollection> muonCollection;
   event.getByLabel(muonTag_, muonCollection);
   if (event.getByLabel(muonTag_, muonCollection)) {
      LogTrace("Analyzer")<<"Reconstructed Muon tracks: " << muonCollection->size() << endl;
   } else {
      LogTrace("") << ">>> Muon collection does not exist !!!";
      return;
   }

   numberOfMuons+=muonCollection->size();
  
   for (unsigned int i=0; i<muonCollection->size(); i++) {
      MuonRef mu(muonCollection,i);
      if (useOnlyGlobalMuons_ && !mu->isGlobalMuon()) continue;
      met_px -= mu->px();
      met_py -= mu->py();
   }

  // Get the MET collection from the event
   Handle<CaloMETCollection> metCollection;
   if (!event.getByLabel(metTag_, metCollection)) {
      LogTrace("") << ">>> MET collection does not exist !!!";
      return;
   }

   CaloMETCollection::const_iterator caloMET = metCollection->begin();
   LogTrace("") << ">>> CaloMET_et, CaloMET_py, CaloMET_py= " << caloMET->et() << ", " << caloMET->px() << ", " << caloMET->py();;
   met_px += caloMET->px();
   met_py += caloMET->py();
   double met_et = sqrt(met_px*met_px+met_py*met_py);
   hMET->Fill(met_et);

  // Get the Jet collection from the event
   Handle<CaloJetCollection> jetCollection;
   if (!event.getByLabel(jetTag_, jetCollection)) {
      LogTrace("") << ">>> CALOJET collection does not exist !!!";
      return;
   }

   CaloJetCollection::const_iterator jet = jetCollection->begin();
   int njets = 0;
   for (jet=jetCollection->begin(); jet!=jetCollection->end(); jet++) {
         if (jet->et()>eJetMin_) njets++;
   }
   hNjets->Fill(njets);
   LogTrace("") << ">>> Number of jets " << jetCollection->size() << "; above " << eJetMin_ << " GeV: " << njets;
   LogTrace("") << ">>> Number of jets above " << eJetMin_ << " GeV: " << njets;

   if (njets>nJetMax_) event_sel = false;

   unsigned int nmuons = 0;
   unsigned int nmuonsForZ = 0;

   hNMu->Fill(muonCollection->size());
   float max_pt = -9999.;
   int i_max_pt = 0;
   for (unsigned int i=0; i<muonCollection->size(); i++) {
      bool muon_sel = true;

      MuonRef mu(muonCollection,i);
      if (useOnlyGlobalMuons_ && !mu->isGlobalMuon()) continue;
      LogTrace("") << "> Processing (global) muon number " << i << "...";
// pt
      double pt = mu->pt();
      hPtMu->Fill(pt);
      LogTrace("") << "\t... pt= " << pt << " GeV";

      if (pt>ptThrForZCount_) nmuonsForZ++;
      if (pt<ptCut_) muon_sel = false;
// eta
      double eta = mu->eta();
      hEtaMu->Fill(eta);
      LogTrace("") << "\t... eta= " << eta;
      if (fabs(eta)>etaCut_) muon_sel = false;

// acoplanarity
      Geom::Phi<double> deltaphi(mu->phi()-atan2(met_py,met_px));
      double acop = deltaphi.value();
      if (acop<0) acop = - acop;
      acop = M_PI - acop;
      hAcop->Fill(acop);
      LogTrace("") << "\t... acop= " << acop;
      if (acop>acopCut_) muon_sel = false;

// transverse mass
      double w_et = mu->pt() + met_et;
      double w_px = mu->px() + met_px;
      double w_py = mu->py() + met_py;
      double massT = w_et*w_et - w_px*w_px - w_py*w_py;
      massT = (massT>0) ? sqrt(massT) : 0;
      hTMass->Fill(massT);
      LogTrace("") << "\t... W_et, W_px, W_py= " << w_et << ", " << w_px << ", " << w_py << " GeV";
      LogTrace("") << "\t... Invariant transverse mass= " << massT << " GeV";
      if (massT<massTMin_) muon_sel = false;
      if (massT>massTMax_) muon_sel = false;

// Isolation
      double ptsum = mu->isolationR03().sumPt;
      hPtSum->Fill(ptsum);
      hPtSumN->Fill(ptsum/pt);
      hTMass_PtSum->Fill(ptsum,massT);
      LogTrace("") << "\t... Isol, Track pt= " << mu->pt() << " GeV, " << " ptsum = " << ptsum;
      if (ptsum/pt > isoCut03_) muon_sel = false;

      if (muon_sel) {
        nmuons++;
        if (pt > max_pt) {  //and identify the highest pt muon
           max_pt = pt;
           i_max_pt = nmuons;
        }
        pt_sel[nmuons] = pt;
        eta_sel[nmuons] = eta;
        acop_sel[nmuons] = acop;
        massT_sel[nmuons] = massT;
        iso_sel[nmuons] = ptsum;
        isoN_sel[nmuons] = ptsum/pt;
      }
   }


      LogTrace("") << "> Muon counts to reject Z= " << nmuonsForZ;
      if (nmuonsForZ>=2) {
            LogTrace("") << ">>>> Event REJECTED";
            event_sel = false;
      }
      LogTrace("") << "> Number of muons for W= " << nmuons;
      if (nmuons<1) {
            LogTrace("") << ">>>> Event REJECTED";
            event_sel = false;
      }

      if (event_sel == true) {
       LogTrace("") << ">>>> Event SELECTED!!!";

// Fill histograms for selected events
       hNMu_sel->Fill(nmuons);
       hMET_sel->Fill(met_et); 
       hNjets_sel->Fill(njets);

// only the combination with highest pt
       hPtMu_sel->Fill(pt_sel[i_max_pt]);
       hEtaMu_sel->Fill(eta_sel[i_max_pt]);
       hAcop_sel->Fill(acop_sel[i_max_pt]);
       hTMass_sel->Fill(massT_sel[i_max_pt]);
       hPtSum_sel->Fill(iso_sel[i_max_pt]);
       hPtSumN_sel->Fill(isoN_sel[i_max_pt]);
       hTMass_PtSum_sel->Fill(iso_sel[i_max_pt],massT_sel[i_max_pt]);

      }

  
}

DEFINE_FWK_MODULE(WMuNuAnalyzer);
