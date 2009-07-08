/** \class WMuNuAnalyzer
 *  Simple analyzer to make some W->MuNu plots
 *
 *  \author M.I. Josa
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "TH1D.h"
#include "TH2D.h"
#include <map>
#include <string>

class WMuNuAnalyzer : public edm::EDFilter {
public:
  WMuNuAnalyzer(const edm::ParameterSet& pset);
  virtual ~WMuNuAnalyzer();
  virtual void beginJob(const edm::EventSetup& eventSetup);
  virtual void endJob();
  virtual bool filter(edm::Event & event, const edm::EventSetup& eventSetup);
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
  double ptThrForZ1_;
  double ptThrForZ2_;
  double acopCut_;
  double eJetMin_;
  int nJetMax_;

// Histograms
  std::map<std::string,TH1D*> h1_;
  std::map<std::string,TH2D*> h2_;
  
  unsigned int numberOfEvents;
  unsigned int numberOfMuons;
};

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/GeometryVector/interface/Phi.h"

using namespace std;
using namespace edm;
using namespace reco;

/// Constructor
WMuNuAnalyzer::WMuNuAnalyzer(const ParameterSet& pset) :
      muonTag_(pset.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_(pset.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met"))),
      jetTag_(pset.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("sisCone5CaloJets"))),
      useOnlyGlobalMuons_(pset.getUntrackedParameter<bool>("UseOnlyGlobalMuons", true)),
      ptCut_(pset.getUntrackedParameter<double>("PtCut", 25.)),
      etaCut_(pset.getUntrackedParameter<double>("EtaCut", 2.1)),
      isRelativeIso_(pset.getUntrackedParameter<bool>("IsRelativeIso", true)),
      isoCut03_(pset.getUntrackedParameter<double>("IsoCut03", 0.1)),
      massTMin_(pset.getUntrackedParameter<double>("MassTMin", 50.)),
      massTMax_(pset.getUntrackedParameter<double>("MassTMax", 200.)),
      ptThrForZ1_(pset.getUntrackedParameter<double>("PtThrForZ1", 20.)),
      ptThrForZ2_(pset.getUntrackedParameter<double>("PtThrForZ2", 10.)),
      acopCut_(pset.getUntrackedParameter<double>("AcopCut", 999999.)),
      eJetMin_(pset.getUntrackedParameter<double>("EJetMin", 999999.)),
      nJetMax_(pset.getUntrackedParameter<int>("NJetMax", 999999))

{
  LogDebug("WMuNuAnalyzer")<<" WMuNuAnalyzer constructor called";
}

/// Destructor
WMuNuAnalyzer::~WMuNuAnalyzer(){
}

void WMuNuAnalyzer::beginJob(const EventSetup& eventSetup){
  // Create output files

  edm::Service<TFileService> fs;

  numberOfEvents = 0;
  numberOfMuons = 0;

  h1_["NMU"]  = fs->make<TH1D>("NMU","Nb. muons in the event",10,0.,10.);
  h1_["PTMU"] = fs->make<TH1D>("PTMU","Pt mu",100,0.,100.);
  h1_["ETAMU"] = fs->make<TH1D>("ETAMU","Eta mu",50,-2.5,2.5);
  h1_["MET"] = fs->make<TH1D>("MET","Missing Transverse Energy (GeV)", 100,0.,200.);
  h1_["TMASS"] = fs->make<TH1D>("TMASS","Rec. Transverse Mass (GeV)",150,0.,300.);
  h1_["ACOP"] = fs->make<TH1D>("ACOP","Mu-MET acoplanarity",50,0.,M_PI);
  h1_["NJETS"] = fs->make<TH1D>("NJETS","njets",25,0.,25.);
  h1_["PTSUM"] = fs->make<TH1D>("PTSUM","Sum pT (GeV)",100,0.,50.);
  h1_["PTSUMN"] = fs->make<TH1D>("PTSUMN","Sum pT/pT",100,0.,50.);
  h2_["TMASS_PTSUM"] = fs->make<TH2D>("TMASS_PTSUM","Rec. Transverse Mass (GeV) vs Sum pT (GeV)",100,0.,50.,150,0.,300.);

  h1_["NMU_SEL"] = fs->make<TH1D>("NMU_SEL","Nb. selected muons",10,0.,10.);
  h1_["PTMU_SEL"] = fs->make<TH1D>("PTMU_SEL","Pt mu",100,0.,100.);
  h1_["ETAMU_SEL"] = fs->make<TH1D>("ETAMU_SEL","Eta mu",50,-2.5,2.5);
  h1_["MET_SEL"] = fs->make<TH1D>("MET_SEL","Missing Transverse Energy (GeV)", 100,0.,200.);
  h1_["TMASS_SEL"] = fs->make<TH1D>("TMASS_SEL","Rec. Transverse Mass (GeV)",150,0.,300.);
  h1_["ACOP_SEL"] = fs->make<TH1D>("ACOP_SEL","Mu-MET acoplanarity",50,0.,M_PI);
  h1_["NJETS_SEL"] = fs->make<TH1D>("NJETS_SEL","njets",25,0.,25.);
  h1_["PTSUM_SEL"] = fs->make<TH1D>("PTSUM_SEL","Sum pT (GeV)",100,0.,50.);
  h1_["PTSUMN_SEL"] = fs->make<TH1D>("PTSUMN_SEL","Sum pT/pT ",100,0.,2.5);
  h2_["TMASS_PTSUM_SEL"] = fs->make<TH2D>("TMASS_PTSUM_SEL","Rec. Transverse Mass (GeV) vs Sum pT (GeV)",100,0.,50.,150,0.,300.);

}

void WMuNuAnalyzer::endJob(){
  LogVerbatim("") << "WMuNuAnalyzer>>> FINAL PRINTOUTS -> BEGIN";
  LogVerbatim("") << "WMuNuAnalyzer>>> Number of analyzed events= " << numberOfEvents;
  LogVerbatim("") << "WMuNuAnalyzer>>> Number of analyzed muons= " << numberOfMuons;

  LogVerbatim("") << "WMuNuAnalyzer>>> FINAL PRINTOUTS -> END";
}
 

bool WMuNuAnalyzer::filter(Event & event, const EventSetup& eventSetup){
  
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
   if (event.getByLabel(muonTag_, muonCollection)) {
      LogTrace("Analyzer")<<"Reconstructed Muon tracks: " << muonCollection->size() << endl;
   } else {
      LogTrace("") << ">>> Muon collection does not exist !!!";
      return false;
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
   if (event.getByLabel(metTag_, metCollection)) {
      LogTrace("Analyzer")<<"CaloMET collection found" << endl;
   } else {
      LogTrace("") << ">>> CaloMET collection does not exist !!!";
      return false;
   }

   CaloMETCollection::const_iterator caloMET = metCollection->begin();
   LogTrace("") << ">>> CaloMET_et, CaloMET_py, CaloMET_py= " << caloMET->et() << ", " << caloMET->px() << ", " << caloMET->py();;
   met_px += caloMET->px();
   met_py += caloMET->py();
   double met_et = sqrt(met_px*met_px+met_py*met_py);
   h1_["MET"]->Fill(met_et);

  // Get the Jet collection from the event
   Handle<CaloJetCollection> jetCollection;
   if (event.getByLabel(jetTag_, jetCollection)) {
      LogTrace("Analyzer")<<"Reconstructed calojets: " << jetCollection->size() << endl;
   } else {
      LogTrace("") << ">>> CALOJET collection does not exist !!!";
      return false;
   }

   CaloJetCollection::const_iterator jet = jetCollection->begin();
   int njets = 0;
   for (jet=jetCollection->begin(); jet!=jetCollection->end(); jet++) {
         if (jet->et()>eJetMin_) njets++;
   }
   h1_["NJETS"]->Fill(njets);
   LogTrace("") << ">>> Number of jets " << jetCollection->size() << "; above " << eJetMin_ << " GeV: " << njets;
   LogTrace("") << ">>> Number of jets above " << eJetMin_ << " GeV: " << njets;

   if (njets>nJetMax_) event_sel = false;

   unsigned int nmuons = 0;
   unsigned int nmuonsForZ1 = 0;
   unsigned int nmuonsForZ2 = 0;

   h1_["NMU"]->Fill(muonCollection->size());
   float max_pt = -9999.;
   int i_max_pt = -1;
   for (unsigned int i=0; i<muonCollection->size(); i++) {
      bool muon_sel = true;

      MuonRef mu(muonCollection,i);
      if (useOnlyGlobalMuons_ && !mu->isGlobalMuon()) continue;
      LogTrace("") << "> Processing (global) muon number " << i << "...";
// pt
      double pt = mu->pt();
      h1_["PTMU"]->Fill(pt);
      LogTrace("") << "\t... pt= " << pt << " GeV";

      if (pt>ptThrForZ1_) nmuonsForZ1++;
      if (pt>ptThrForZ2_) nmuonsForZ2++;
      if (pt<ptCut_) muon_sel = false;
// eta
      double eta = mu->eta();
      h1_["ETAMU"]->Fill(eta);
      LogTrace("") << "\t... eta= " << eta;
      if (fabs(eta)>etaCut_) muon_sel = false;

// acoplanarity
      Geom::Phi<double> deltaphi(mu->phi()-atan2(met_py,met_px));
      double acop = deltaphi.value();
      if (acop<0) acop = - acop;
      acop = M_PI - acop;
      h1_["ACOP"]->Fill(acop);
      LogTrace("") << "\t... acop= " << acop;
      if (acop>acopCut_) muon_sel = false;

// transverse mass
      double w_et = mu->pt() + met_et;
      double w_px = mu->px() + met_px;
      double w_py = mu->py() + met_py;
      double massT = w_et*w_et - w_px*w_px - w_py*w_py;
      massT = (massT>0) ? sqrt(massT) : 0;
      h1_["TMASS"]->Fill(massT);
      LogTrace("") << "\t... W_et, W_px, W_py= " << w_et << ", " << w_px << ", " << w_py << " GeV";
      LogTrace("") << "\t... Invariant transverse mass= " << massT << " GeV";
      if (massT<massTMin_) muon_sel = false;
      if (massT>massTMax_) muon_sel = false;

// Isolation
      double ptsum = mu->isolationR03().sumPt;
      h1_["PTSUM"]->Fill(ptsum);
      h1_["PTSUMN"]->Fill(ptsum/pt);
      h2_["TMASS_PTSUM"]->Fill(ptsum,massT);
      LogTrace("") << "\t... Isol, Track pt= " << mu->pt() << " GeV, " << " ptsum = " << ptsum;
      if (ptsum/pt > isoCut03_) muon_sel = false;

      if (muon_sel && nmuons<5) {
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
        nmuons++;
      }
   }


      LogTrace("") << "> Muon counts to reject Z= " << nmuonsForZ1 << ", " << nmuonsForZ2;
      if (nmuonsForZ1>=1 && nmuonsForZ2>=2) {
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
       h1_["NMU_SEL"]->Fill(nmuons);
       h1_["MET_SEL"]->Fill(met_et); 
       h1_["NJETS_SEL"]->Fill(njets);

// only the combination with highest pt
       h1_["PTMU_SEL"]->Fill(pt_sel[i_max_pt]);
       h1_["ETAMU_SEL"]->Fill(eta_sel[i_max_pt]);
       h1_["ACOP_SEL"]->Fill(acop_sel[i_max_pt]);
       h1_["TMASS_SEL"]->Fill(massT_sel[i_max_pt]);
       h1_["PTSUM_SEL"]->Fill(iso_sel[i_max_pt]);
       h1_["PTSUMN_SEL"]->Fill(isoN_sel[i_max_pt]);
       h2_["TMASS_PTSUM_SEL"]->Fill(iso_sel[i_max_pt],massT_sel[i_max_pt]);

      }

      return event_sel;
  
}

DEFINE_FWK_MODULE(WMuNuAnalyzer);
