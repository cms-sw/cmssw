#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "TH1D.h"
#include <map>

#ifdef IS_PAT
class WMuNuPATSelector : public edm::EDFilter {
#else
class WMuNuAODSelector : public edm::EDFilter {
#endif
public:
#ifdef IS_PAT
  WMuNuPATSelector (const edm::ParameterSet &);
#else
  WMuNuAODSelector (const edm::ParameterSet &);
#endif
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();
private:
  edm::InputTag trigTag_;
  edm::InputTag muonTag_;
  edm::InputTag metTag_;
  bool metIncludesMuons_;
  edm::InputTag jetTag_;

  const std::string muonTrig_;
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

  std::map<std::string,TH1D*> h1_;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#ifdef IS_PAT
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#else
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"
#endif

#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/Common/interface/View.h"
  
using namespace edm;
using namespace std;
#ifdef IS_PAT
      using namespace pat;
#else
      using namespace reco;
#endif

#ifdef IS_PAT
WMuNuPATSelector::WMuNuPATSelector( const ParameterSet & cfg ) :
#else
WMuNuAODSelector::WMuNuAODSelector( const ParameterSet & cfg ) :
#endif
      // Input collections
      trigTag_(cfg.getUntrackedParameter<edm::InputTag> ("TrigTag", edm::InputTag("TriggerResults::HLT"))),
      muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met"))),
      metIncludesMuons_(cfg.getUntrackedParameter<bool> ("METIncludesMuons", false)),
      jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("sisCone5CaloJets"))),

      // Main cuts 
      muonTrig_(cfg.getUntrackedParameter<std::string> ("MuonTrig", "HLT_Mu9")),
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

      // Z rejection
      ptThrForZ1_(cfg.getUntrackedParameter<double>("PtThrForZ1", 20.)),
      ptThrForZ2_(cfg.getUntrackedParameter<double>("PtThrForZ2", 10.)),

      // Top rejection
      eJetMin_(cfg.getUntrackedParameter<double>("EJetMin", 999999.)),
      nJetMax_(cfg.getUntrackedParameter<int>("NJetMax", 999999))
{
}

#ifdef IS_PAT
void WMuNuPATSelector::beginJob(const EventSetup &) {
#else
void WMuNuAODSelector::beginJob(const EventSetup &) {
#endif
      nall = 0;
      nrec = 0;
      niso = 0;
      nhlt = 0;
      nmet = 0;
      nsel = 0;

      edm::Service<TFileService> fs;
      char chtitle[256] = "";

      snprintf(chtitle, 255, "Trigger response (bit %s)", muonTrig_.data());
      h1_["TRIG"]  = fs->make<TH1D>("TRIG",chtitle,2,-0.5,1.5);
      h1_["PT"] = fs->make<TH1D>("PT","Muon transverse momentum [GeV]",100,0.,100.);
      h1_["ETA"] = fs->make<TH1D>("ETA","Muon pseudorapidity",50,-2.5,2.5);
      h1_["DXY"] = fs->make<TH1D>("DXY","Muon transverse distance to beam spot [cm]",100,-1.,1.);
      h1_["CHI2"] = fs->make<TH1D>("CHI2","Normalized Chi2, inner track fit",100,0.,100.);
      h1_["NHITS"] = fs->make<TH1D>("NHITS","Number of hits in inner track",35,-0.5,34.5);
      h1_["TKMU"] = fs->make<TH1D>("TKMU","Tracker Muon flag (for global muons)",2,-0.5,1.5);
      h1_["MET"] = fs->make<TH1D>("MET","Missing Transverse Energy [GeV]", 100,0.,200.);
      h1_["MT"] = fs->make<TH1D>("MT","Transverse Mass [GeV]",150,0.,300.);
      h1_["ACOP"] = fs->make<TH1D>("ACOP","Mu-MET acoplanarity",50,0.,M_PI);

      snprintf(chtitle, 255, "Number of jets above %.2f GeV", eJetMin_);
      h1_["NJETS"] = fs->make<TH1D>("NJETS",chtitle,25,-0.5,24.5);
      if (isRelativeIso_) {
            h1_["ISO"] = fs->make<TH1D>("ISO","Relative isolation variable", 100, 0., 1.);
      } else {
            h1_["ISO"] = fs->make<TH1D>("ISO","Transverse energy/momentum in isolation cone (GeV)", 100, 0., 50.);
      }

      snprintf(chtitle, 255, "Z rejetion: number of muons above %.2f GeV", ptThrForZ1_);
      h1_["NZ1"] = fs->make<TH1D>("NZ1",chtitle, 10, -0.5, 9.5);
      snprintf(chtitle, 255, "Z rejetion: number of muons above %.2f GeV", ptThrForZ2_);
      h1_["NZ2"] = fs->make<TH1D>("NZ2",chtitle, 10, -0.5, 9.5);
}

#ifdef IS_PAT
void WMuNuPATSelector::endJob() {
#else
void WMuNuAODSelector::endJob() {
#endif
      double all = nall;
      double erec = nrec/all;
      double eiso = niso/all;
      double ehlt = nhlt/all;
      double emet = nmet/all;
      double esel = nsel/all;
      LogVerbatim("") << "\n>>>>>> W SELECTION SUMMARY BEGIN >>>>>>>>>>>>>>>";
      LogVerbatim("") << "Total numer of events analyzed: " << nall << " [events]";

      double num = nrec;
      double eff = erec;
      double err = sqrt(eff*(1-eff)/all);
      double effstep = 1.;
      double errstep = 0.;
      LogVerbatim("") << "Passing Pt/Eta/Quality cuts:    " << num << " [events], (" << setprecision(4) << eff*100. <<" +/- "<< setprecision(2) << err*100. << ")%";

      num = niso;
      eff = eiso;
      err = sqrt(eff*(1-eff)/all);
      effstep = eiso/erec;
      errstep = sqrt(effstep*(1-effstep)/nrec);
      LogVerbatim("") << "Passing isolation cuts:         " << num << " [events], (" << setprecision(4) << eff*100. <<" +/- "<< setprecision(2) << err*100. << ")%, to previous step: (" <<  setprecision(4) << effstep*100. << " +/- "<< setprecision(2) << errstep*100. <<")%";

      num = nhlt;
      eff = ehlt;
      err = sqrt(eff*(1-eff)/all);
      effstep = ehlt/eiso;
      errstep = sqrt(effstep*(1-effstep)/niso);
      LogVerbatim("") << "Passing HLT criteria:           " << num << " [events], (" << setprecision(4) << eff*100. <<" +/- "<< setprecision(2) << err*100. << ")%, to previous step: (" <<  setprecision(4) << effstep*100. << " +/- "<< setprecision(2) << errstep*100. <<")%";

      num = nmet;
      eff = emet;
      err = sqrt(eff*(1-eff)/all);
      effstep = emet/ehlt;
      errstep = sqrt(effstep*(1-effstep)/nhlt);
      LogVerbatim("") << "Passing MET/acoplanarity cuts:  " << num << " [events], (" << setprecision(4) << eff*100. <<" +/- "<< setprecision(2) << err*100. << ")%, to previous step: (" <<  setprecision(4) << effstep*100. << " +/- "<< setprecision(2) << errstep*100. <<")%";

      num = nsel;
      eff = esel;
      err = sqrt(eff*(1-eff)/all);
      effstep = esel/emet;
      errstep = sqrt(effstep*(1-effstep)/nmet);
      LogVerbatim("") << "Passing Z/top rejection cuts:   " << num << " [events], (" << setprecision(4) << eff*100. <<" +/- "<< setprecision(2) << err*100. << ")%, to previous step: (" <<  setprecision(4) << effstep*100. << " +/- "<< setprecision(2) << errstep*100. <<")%";

      LogVerbatim("") << ">>>>>> W SELECTION SUMMARY END   >>>>>>>>>>>>>>>\n";
}

#ifdef IS_PAT
bool WMuNuPATSelector::filter (Event & ev, const EventSetup &) {
#else
bool WMuNuAODSelector::filter (Event & ev, const EventSetup &) {
#endif

      // Reset global event selection flags
      bool rec_sel = false;
      bool iso_sel = false;
      bool hlt_sel = false;
      bool met_sel = false;
      bool sel = false;

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
      h1_["MET"]->Fill(met_et);

      // Jet collection
      Handle<View<Jet> > jetCollection;
      if (!ev.getByLabel(jetTag_, jetCollection)) {
            LogError("") << ">>> JET collection does not exist !!!";
            return false;
      }
      unsigned int jetCollectionSize = jetCollection->size();
      int njets = 0;
      for (unsigned int i=0; i<jetCollectionSize; i++) {
            const Jet& jet = jetCollection->at(i);
            if (jet.et()>eJetMin_) njets++;
      }
      LogTrace("") << ">>> Total number of jets: " << jetCollectionSize;
      LogTrace("") << ">>> Number of jets above " << eJetMin_ << " [GeV]: " << njets;
      h1_["NJETS"]->Fill(njets);

      // Trigger
      Handle<TriggerResults> triggerResults;
      TriggerNames trigNames;
      if (!ev.getByLabel(trigTag_, triggerResults)) {
            LogError("") << ">>> TRIGGER collection does not exist !!!";
            return false;
      }
      trigNames.init(*triggerResults);
      /*
      for (unsigned int i=0; i<triggerResults->size(); i++) {
            if (triggerResults->accept(i)) {
                  LogError("") << "Accept by: " << i << ", Trigger: " << trigNames.triggerName(i);
            }
      }
      */
      bool trigger_fired = false;
      int itrig1 = trigNames.triggerIndex(muonTrig_);
      if (triggerResults->accept(itrig1)) trigger_fired = true;
      LogTrace("") << ">>> Trigger bit: " << trigger_fired << " (" << muonTrig_ << ")";
      h1_["TRIG"]->Fill((double)trigger_fired);

      nall++;

      // Loop to reject/control Z->mumu is done separately
      unsigned int nmuonsForZ1 = 0;
      unsigned int nmuonsForZ2 = 0;
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            double pt = mu.pt();
            if (useTrackerPt_) {
                  reco::TrackRef tk = mu.innerTrack();
                  if (mu.innerTrack().isNull()) continue;
                  pt = tk->pt();
            }
            if (pt>ptThrForZ1_) nmuonsForZ1++;
            if (pt>ptThrForZ2_) nmuonsForZ2++;
      }
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ1_ << " [GeV]: " << nmuonsForZ1;
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ2_ << " [GeV]: " << nmuonsForZ2;
      h1_["NZ1"]->Fill((double)nmuonsForZ1);
      h1_["NZ2"]->Fill((double)nmuonsForZ2);

      // Central W->mu nu selection criteria
      const int NFLAGS = 10;
      bool muon_sel[NFLAGS];
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            for (int j=0; j<NFLAGS; ++j) muon_sel[j] = false;
            bool rec_sel_this = true;
            bool iso_sel_this = true;
            bool met_sel_this = true;

            const Muon& mu = muonCollection->at(i);
            if (!mu.isGlobalMuon()) continue;
            if (mu.globalTrack().isNull()) continue;
            if (mu.innerTrack().isNull()) continue;

            LogTrace("") << "> Wsel: processing muon number " << i << "...";
            reco::TrackRef gm = mu.globalTrack();
            reco::TrackRef tk = mu.innerTrack();

            // Pt,eta cuts
            double pt = mu.pt();
            if (useTrackerPt_) pt = tk->pt();
            double eta = mu.eta();
            LogTrace("") << "\t... pt, eta: " << pt << " [GeV], " << eta;;
            if (pt>ptCut_) muon_sel[0] = true; else rec_sel_this = false;
            if (fabs(eta)<etaCut_) muon_sel[1] = true; else rec_sel_this = false;

            // d0, chi2, nhits quality cuts
            double dxy = tk->dxy(beamSpotHandle->position());
            double normalizedChi2 = gm->normalizedChi2();
            double trackerHits = tk->numberOfValidHits();
            LogTrace("") << "\t... dxy, normalizedChi2, trackerHits, isTrackerMuon?: " << dxy << " [cm], " << normalizedChi2 << ", " << trackerHits << ", " << mu.isTrackerMuon();
            if (fabs(dxy)<dxyCut_) muon_sel[2] = true; else rec_sel_this = false;
            if (normalizedChi2<normalizedChi2Cut_) muon_sel[3] = true; else rec_sel_this = false;
            if (trackerHits>=trackerHitsCut_) muon_sel[4] = true; else rec_sel_this = false;
            if (mu.isTrackerMuon()) muon_sel[5] = true; else rec_sel_this = false;

            h1_["PT"]->Fill(pt);
            h1_["ETA"]->Fill(eta);
            h1_["DXY"]->Fill(dxy);
            h1_["CHI2"]->Fill(normalizedChi2);
            h1_["NHITS"]->Fill(trackerHits);
            h1_["TKMU"]->Fill((double)mu.isTrackerMuon());

            // "rec" => pt,eta and wuality cuts are satisfied
            if (rec_sel_this) rec_sel = true;

            // Isolation cuts
            double isovar = mu.isolationR03().sumPt;
            if (isCombinedIso_) {
                  isovar += mu.isolationR03().emEt;
                  isovar += mu.isolationR03().hadEt;
            }
            if (isRelativeIso_) isovar /= pt;
            if (isovar<isoCut03_) muon_sel[6] = true; else iso_sel_this = false;
            LogTrace("") << "\t... isolation value" << isovar <<", isolated? " << muon_sel[6];
            h1_["ISO"]->Fill(isovar);

            // "iso" => "rec" AND "muon is isolated"
            if (rec_sel_this && iso_sel_this) iso_sel = true;

            // "hlt" => "rec" AND "iso" AND "event is triggered"
            if (rec_sel_this && iso_sel_this && trigger_fired) hlt_sel = true;

            // MET/MT cuts
            double w_et = met_et;
            double w_px = met_px;
            double w_py = met_py;
            if (useTrackerPt_) {
                  w_et += tk->pt();
                  w_px += tk->px();
                  w_py += tk->py();
            } else {
                  w_et += mu.pt();
                  w_px += mu.px();
                  w_py += mu.py();
            }
            double massT = w_et*w_et - w_px*w_px - w_py*w_py;
            massT = (massT>0) ? sqrt(massT) : 0;

            LogTrace("") << "\t... W mass, W_et, W_px, W_py: " << massT << ", " << w_et << ", " << w_px << ", " << w_py << " [GeV]";
            if (met_et>metMin_ && met_et<metMax_) muon_sel[7] = true; else met_sel_this = false;
            if (massT>mtMin_ && massT<mtMax_) muon_sel[8] = true; else met_sel_this = false;
            h1_["MT"]->Fill(massT);

            // Acoplanarity cuts
            Geom::Phi<double> deltaphi(mu.phi()-atan2(met_py,met_px));
            double acop = deltaphi.value();
            if (acop<0) acop = - acop;
            acop = M_PI - acop;
            LogTrace("") << "\t... acoplanarity: " << acop;
            if (acop<acopCut_) muon_sel[9] = true; else met_sel_this = false;
            h1_["ACOP"]->Fill(acop);

            // "met" => "rec" AND "iso" AND "hlt" AND "passes MET/MT and acoplanarity cuts"
            if (rec_sel_this && iso_sel_this && trigger_fired && met_sel_this) met_sel = true;
      }

      // Collect final flags
      if (rec_sel) nrec++;
      if (iso_sel) niso++;
      if (hlt_sel) nhlt++;
      if (met_sel) nmet++;

      if (met_sel && (nmuonsForZ1<1||nmuonsForZ2<2) && njets<nJetMax_) {
            sel = true;
            nsel++;
      }

      if (sel) {
            LogTrace("") << ">>>> Event ACCEPTED";
      } else {
            LogTrace("") << ">>>> Event REJECTED";
      }

      return sel;

}

#include "FWCore/Framework/interface/MakerMacros.h"

#ifdef IS_PAT
      DEFINE_FWK_MODULE( WMuNuPATSelector );
#else
      DEFINE_FWK_MODULE( WMuNuAODSelector );
#endif
