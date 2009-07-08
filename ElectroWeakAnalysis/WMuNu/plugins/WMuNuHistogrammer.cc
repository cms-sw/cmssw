/** \class WMuNuHistogrammer
 *  Simple histogrammer to make some W->MuNu plots
 *
 *  \author M.I. Josa
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "TH1D.h"
#include "TH2D.h"
#include <map>
#include <string>

class WMuNuHistogrammer : public edm::EDFilter {
public:
  WMuNuHistogrammer(const edm::ParameterSet& pset);
  virtual ~WMuNuHistogrammer();
  virtual void beginJob(const edm::EventSetup& eventSetup);
  virtual void endJob();
  virtual bool filter(edm::Event & event, const edm::EventSetup& eventSetup);
private:
  edm::InputTag trigTag_;
  edm::InputTag muonTag_;
  edm::InputTag metTag_;
  bool metIncludesMuons_;
  edm::InputTag jetTag_;

  const std::string muonTrig_;
  bool useOnlyGlobalMuons_;
  double ptCut_;
  double etaCut_;
  bool isRelativeIso_;
  bool isCombinedIso_;
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
  unsigned int numberOfSelectedEvents;
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

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/View.h"

using namespace std;
using namespace edm;
using namespace reco;

/// Constructor
WMuNuHistogrammer::WMuNuHistogrammer(const ParameterSet& cfg) :
      trigTag_(cfg.getUntrackedParameter<edm::InputTag> ("TrigTag", edm::InputTag("TriggerResults::HLT"))),
      muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met"))),
      metIncludesMuons_(cfg.getUntrackedParameter<bool> ("METIncludesMuons", false)),
      jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("sisCone5CaloJets"))),

      muonTrig_(cfg.getUntrackedParameter<std::string> ("MuonTrig", "HLT_Mu9")),
      useOnlyGlobalMuons_(cfg.getUntrackedParameter<bool>("UseOnlyGlobalMuons", true)),
      ptCut_(cfg.getUntrackedParameter<double>("PtCut", 25.)),
      etaCut_(cfg.getUntrackedParameter<double>("EtaCut", 2.1)),
      isRelativeIso_(cfg.getUntrackedParameter<bool>("IsRelativeIso", true)),
      isCombinedIso_(cfg.getUntrackedParameter<bool>("IsCombinedIso", false)),
      isoCut03_(cfg.getUntrackedParameter<double>("IsoCut03", 0.1)),
      massTMin_(cfg.getUntrackedParameter<double>("MassTMin", 50.)),
      massTMax_(cfg.getUntrackedParameter<double>("MassTMax", 200.)),
      ptThrForZ1_(cfg.getUntrackedParameter<double>("PtThrForZ1", 20.)),
      ptThrForZ2_(cfg.getUntrackedParameter<double>("PtThrForZ2", 10.)),
      acopCut_(cfg.getUntrackedParameter<double>("AcopCut", 999999.)),
      eJetMin_(cfg.getUntrackedParameter<double>("EJetMin", 999999.)),
      nJetMax_(cfg.getUntrackedParameter<int>("NJetMax", 999999))

{
  LogDebug("WMuNuHistogrammer")<<" WMuNuHistogrammer constructor called";
}

/// Destructor
WMuNuHistogrammer::~WMuNuHistogrammer(){
}

void WMuNuHistogrammer::beginJob(const EventSetup& eventSetup){
  // Create output files

  edm::Service<TFileService> fs;

  numberOfEvents = 0;
  numberOfSelectedEvents = 0;
  numberOfMuons = 0;

  h1_["TRIG"]  = fs->make<TH1D>("TRIG","Trigger flag",2,-0.5,1.5);
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

  h1_["TRIG_SEL"]  = fs->make<TH1D>("TRIG_SEL","Trigger flag",2,-0.5,1.5);
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

void WMuNuHistogrammer::endJob(){
  LogVerbatim("") << "WMuNuHistogrammer>>> FINAL PRINTOUTS -> BEGIN";
  LogVerbatim("") << "WMuNuHistogrammer>>> Number of analyzed events= " << numberOfEvents;
  LogVerbatim("") << "WMuNuHistogrammer>>> Number of analyzed muons= " << numberOfMuons;
  LogVerbatim("") << "WMuNuHistogrammer>>> Number of selected events= " << numberOfSelectedEvents;
  LogVerbatim("") << "WMuNuHistogrammer>>> FINAL PRINTOUTS -> END";
}
 

bool WMuNuHistogrammer::filter(Event & event, const EventSetup& eventSetup){
  
   numberOfEvents++;

   std::vector<double> pt_sel;
   std::vector<double> eta_sel;
   std::vector<double> acop_sel;
   std::vector<double> massT_sel;
   std::vector<double> iso_sel;
   std::vector<double> isoN_sel;

   bool cut_sel = true;
  
   double met_px = 0.;
   double met_py = 0.;

  // Get the Muon Track collection from the event
   Handle<View<Muon> > muonCollection;
   if (event.getByLabel(muonTag_, muonCollection)) {
      LogTrace("Histogrammer")<<"Reconstructed Muon tracks: " << muonCollection->size() << endl;
   } else {
      LogTrace("") << ">>> Muon collection does not exist !!!";
      return false;
   }
   unsigned int muonCollectionSize = muonCollection->size();
   numberOfMuons += muonCollectionSize;
  
  // Get the MET collection from the event
   Handle<View<MET> > metCollection;
   if (event.getByLabel(metTag_, metCollection)) {
      LogTrace("Histogrammer")<<"CaloMET collection found" << endl;
   } else {
      LogTrace("") << ">>> CaloMET collection does not exist !!!";
      return false;
   }

   const MET& met = metCollection->at(0);
   met_px = met.px();
   met_py = met.py();
   if (!metIncludesMuons_) {
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            const Muon& mu = muonCollection->at(i);
            if (useOnlyGlobalMuons_ && !mu.isGlobalMuon()) continue;
            met_px -= mu.px();
            met_py -= mu.py();
      }
   }
   double met_et = sqrt(met_px*met_px+met_py*met_py);
   LogTrace("") << ">>> MET, MET_px, MET_py= " << met_et << ", " << met_px << ", " << met_py;
   h1_["MET"]->Fill(met_et);

  // Get the Jet collection from the event
   Handle<View<Jet> > jetCollection;
   if (event.getByLabel(jetTag_, jetCollection)) {
      LogTrace("Histogrammer")<<"Reconstructed calojets: " << jetCollection->size() << endl;
   } else {
      LogTrace("") << ">>> CALOJET collection does not exist !!!";
      return false;
   }
   unsigned int jetCollectionSize = jetCollection->size();
   int njets = 0;
   for (unsigned int i=0; i<jetCollectionSize; i++) {
      const Jet& jet = jetCollection->at(i);
      if (jet.et()>eJetMin_) njets++;
   }
   LogTrace("") << ">>> Total number of jets= " << jetCollectionSize;
   LogTrace("") << ">>> Number of jets above " << eJetMin_ << " GeV: " << njets;
   h1_["NJETS"]->Fill(njets);

   if (njets>nJetMax_) cut_sel = false;

   // Trigger
   Handle<TriggerResults> triggerResults;
   TriggerNames trigNames;
   if (!event.getByLabel(trigTag_, triggerResults)) {
            LogError("") << ">>> TRIGGER collection does not exist !!!";
            return false;
   }
   
   bool trigger_sel = true;
   trigNames.init(*triggerResults);
   int itrig1 = trigNames.triggerIndex(muonTrig_);
   if (!triggerResults->accept(itrig1)) trigger_sel = false;
   h1_["TRIG"]->Fill((double)trigger_sel);

   unsigned int nmuonsForZ1 = 0;
   unsigned int nmuonsForZ2 = 0;

   h1_["NMU"]->Fill(muonCollectionSize);
   float max_pt = -9999.;
   int i_max_pt = -1;
   for (unsigned int i=0; i<muonCollectionSize; i++) {
      bool muon_sel = true;

      const Muon& mu = muonCollection->at(i);
      if (useOnlyGlobalMuons_ && !mu.isGlobalMuon()) continue;
      if (mu.innerTrack().isNull()) continue;
      TrackRef tk = mu.innerTrack();
      LogTrace("") << "> Processing (global) muon number " << i << "...";
// pt
      double pt = tk->pt();
      h1_["PTMU"]->Fill(pt);
      LogTrace("") << "\t... pt= " << pt << " GeV";

      if (pt>ptThrForZ1_) nmuonsForZ1++;
      if (pt>ptThrForZ2_) nmuonsForZ2++;
      if (pt<ptCut_) muon_sel = false;
// eta
      double eta = tk->eta();
      h1_["ETAMU"]->Fill(eta);
      LogTrace("") << "\t... eta= " << eta;
      if (fabs(eta)>etaCut_) muon_sel = false;

// acoplanarity
      Geom::Phi<double> deltaphi(tk->phi()-atan2(met_py,met_px));
      double acop = deltaphi.value();
      if (acop<0) acop = - acop;
      acop = M_PI - acop;
      h1_["ACOP"]->Fill(acop);
      LogTrace("") << "\t... acop= " << acop;
      if (acop>acopCut_) muon_sel = false;

// transverse mass
      double w_et = tk->pt() + met_et;
      double w_px = tk->px() + met_px;
      double w_py = tk->py() + met_py;
      double massT = w_et*w_et - w_px*w_px - w_py*w_py;
      massT = (massT>0) ? sqrt(massT) : 0;
      h1_["TMASS"]->Fill(massT);
      LogTrace("") << "\t... W_et, W_px, W_py= " << w_et << ", " << w_px << ", " << w_py << " GeV";
      LogTrace("") << "\t... Invariant transverse mass= " << massT << " GeV";
      if (massT<massTMin_) muon_sel = false;
      if (massT>massTMax_) muon_sel = false;

// Isolation
      bool iso = false;
      double etsum = mu.isolationR03().sumPt;
      if (isCombinedIso_) {
            etsum += mu.isolationR03().emEt;
            etsum += mu.isolationR03().hadEt;
      }
      if (isRelativeIso_) {
            if (etsum/pt<isoCut03_) iso=true;
      } else {
            if (etsum<isoCut03_) iso=true;
      }
      h1_["PTSUM"]->Fill(etsum);
      h1_["PTSUMN"]->Fill(etsum/pt);
      h2_["TMASS_PTSUM"]->Fill(etsum,massT);
      LogTrace("") << "\t... Isol, Track pt= " << pt << " GeV, " << " etsum = " << etsum;
      if (!iso) muon_sel = false;

      if (muon_sel) {
        if (pt > max_pt) {  //and identify the highest pt muon
           max_pt = pt;
           i_max_pt = pt_sel.size();
        }
        pt_sel.push_back(pt);
        eta_sel.push_back(eta);
        acop_sel.push_back(acop);
        massT_sel.push_back(massT);
        iso_sel.push_back(etsum);
        isoN_sel.push_back(etsum/pt);
      }
   }

      int nmuons = pt_sel.size();
      LogTrace("") << "> Muon counts to reject Z= " << nmuonsForZ1 << ", " << nmuonsForZ2;
      if (nmuonsForZ1>=1 && nmuonsForZ2>=2) {
            LogTrace("") << ">>>> Event REJECTED";
            cut_sel = false;
      }
      LogTrace("") << "> Number of muons for W= " << nmuons;
      if (nmuons<1) {
            LogTrace("") << ">>>> Event REJECTED";
            cut_sel = false;
      }

      if (cut_sel == true) h1_["TRIG_SEL"]->Fill((double)trigger_sel);
      bool event_sel = cut_sel && trigger_sel;

      if (event_sel) {
       LogTrace("") << ">>>> Event SELECTED!!!";
       numberOfSelectedEvents++;

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

DEFINE_FWK_MODULE(WMuNuHistogrammer);
