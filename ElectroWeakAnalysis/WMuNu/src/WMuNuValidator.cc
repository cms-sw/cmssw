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
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "TH1D.h"
#include <map>

class WMuNuValidator : public edm::EDFilter {
public:
  WMuNuValidator (const edm::ParameterSet &);
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void beginJob() override;
  virtual void endJob() override;
  void init_histograms();
  void fill_histogram(const char*, const double&);
private:
  bool fastOption_;
  edm::EDGetTokenT<edm::TriggerResults> trigToken_;
  edm::EDGetTokenT<edm::View<reco::Muon> > muonToken_;
  edm::InputTag metTag_;
  edm::EDGetTokenT<edm::View<reco::MET> > metToken_;
  bool metIncludesMuons_;
  edm::InputTag jetTag_;
  edm::EDGetTokenT<edm::View<reco::Jet> > jetToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const std::string muonTrig_;
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
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Common/interface/TriggerNames.h"

using namespace edm;
using namespace std;
using namespace reco;

WMuNuValidator::WMuNuValidator( const ParameterSet & cfg ) :
      // Fast selection (no histograms or book-keeping)
      fastOption_(cfg.getUntrackedParameter<bool> ("FastOption", false)),

      // Input collections
      trigToken_(consumes<TriggerResults>(cfg.getUntrackedParameter<edm::InputTag> ("TrigTag", edm::InputTag("TriggerResults::HLT")))),
      muonToken_(consumes<View<Muon> >(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons")))),
      metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met"))),
      metToken_(consumes<View<MET> >(metTag_)),
      metIncludesMuons_(cfg.getUntrackedParameter<bool> ("METIncludesMuons", false)),
      jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("sisCone5CaloJets"))),
      jetToken_(consumes<View<Jet> >(jetTag_)),
      beamSpotToken_(consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"))),

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

void WMuNuValidator::beginJob() {
      nall = 0;
      nsel = 0;

      if (!fastOption_) {
            nrec = 0;
            niso = 0;
            nhlt = 0;
            nmet = 0;
            init_histograms();
      }
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
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,50,-2.5,2.5);

            snprintf(chname, 255, "DXY%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Muon transverse distance to beam spot [cm]");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,100,-0.5,0.5);

            snprintf(chname, 255, "CHI2%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Normalized Chi2, inner track fit");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,100,0.,100.);

            snprintf(chname, 255, "NHITS%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Number of hits, inner track");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,40,-0.5,39.5);

            snprintf(chname, 255, "ValidMuonHits%s", chsuffix[i].data());
            snprintf(chtitle, 255, "number Of Valid Muon Hits");
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,40,-0.5,39.5);

            snprintf(chname, 255, "TKMU%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Tracker-muon flag (for global muons)");
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
            snprintf(chtitle, 255, "Trigger response (bit %s)", muonTrig_.data());
            h1_[chname]  = subDir[i]->make<TH1D>(chname,chtitle,2,-0.5,1.5);

            snprintf(chname, 255, "MT%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Transverse mass (%s) [GeV]", metTag_.label().data());
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,150,0.,300.);

            snprintf(chname, 255, "MET%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Missing transverse energy (%s) [GeV]", metTag_.label().data());
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,100,0.,200.);

            snprintf(chname, 255, "ACOP%s", chsuffix[i].data());
            snprintf(chtitle, 255, "MU-MET (%s) acoplanarity", metTag_.label().data());
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,50,0.,M_PI);

            snprintf(chname, 255, "NZ1%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Z rejection: number of muons above %.2f GeV", ptThrForZ1_);
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle, 10, -0.5, 9.5);

            snprintf(chname, 255, "NZ2%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Z rejection: number of muons above %.2f GeV", ptThrForZ2_);
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle, 10, -0.5, 9.5);

            snprintf(chname, 255, "NJETS%s", chsuffix[i].data());
            snprintf(chtitle, 255, "Number of jets (%s) above %.2f GeV", jetTag_.label().data(), eJetMin_);
            h1_[chname] = subDir[i]->make<TH1D>(chname,chtitle,10,-0.5,9.5);

      }
}

void WMuNuValidator::fill_histogram(const char* name, const double& var) {
      if (fastOption_) return;
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
        double eiso = niso/all;
        double ehlt = nhlt/all;
        double emet = nmet/all;

        double num = nrec;
        double eff = erec;
        double err = sqrt(eff*(1-eff)/all);
        LogVerbatim("") << "Passing Pt/Eta/Quality cuts:    " << num << " [events], (" << setprecision(4) << eff*100. <<" +/- "<< setprecision(2) << err*100. << ")%";

        num = niso;
        eff = eiso;
        err = sqrt(eff*(1-eff)/all);
        double effstep = 0.;
        double errstep = 0.;
        if (nrec>0) effstep = eiso/erec;
        if (nrec>0) errstep = sqrt(effstep*(1-effstep)/nrec);
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
      bool iso_sel = false;
      bool hlt_sel = false;
      bool met_sel = false;
      bool all_sel = false;

      // Muon collection
      Handle<View<Muon> > muonCollection;
      if (!ev.getByToken(muonToken_, muonCollection)) {
            LogError("") << ">>> Muon collection does not exist !!!";
            return 0;
      }
      unsigned int muonCollectionSize = muonCollection->size();

      // Beam spot
      Handle<reco::BeamSpot> beamSpotHandle;
      if (!ev.getByToken(beamSpotToken_, beamSpotHandle)) {
            LogTrace("") << ">>> No beam spot found !!!";
            return false;
      }

      // MET
      double met_px = 0.;
      double met_py = 0.;
      Handle<View<MET> > metCollection;
      if (!ev.getByToken(metToken_, metCollection)) {
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
      fill_histogram("MET_BEFORECUTS",met_et);

      // Trigger
      Handle<TriggerResults> triggerResults;
      if (!ev.getByToken(trigToken_, triggerResults)) {
            LogError("") << ">>> TRIGGER collection does not exist !!!";
            return 0;
      }
      const edm::TriggerNames & triggerNames = ev.triggerNames(*triggerResults);
    /*
      for (unsigned int i=0; i<triggerResults->size(); i++) {
            if (triggerResults->accept(i)) {
                  LogTrace("") << "Accept by: " << i << ", Trigger: " << triggerNames.triggerName(i);
            }
      }
    */
      bool trigger_fired = false;
      int itrig1 = triggerNames.triggerIndex(muonTrig_);
      if (triggerResults->accept(itrig1)) trigger_fired = true;
      LogTrace("") << ">>> Trigger bit: " << trigger_fired << " (" << muonTrig_ << ")";
      fill_histogram("TRIG_BEFORECUTS",trigger_fired);

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
      fill_histogram("NZ1_BEFORECUTS",nmuonsForZ1);
      fill_histogram("NZ2_BEFORECUTS",nmuonsForZ2);

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
      fill_histogram("NJETS_BEFORECUTS",njets);

      // Start counting, reject already events if possible (under FastOption flag)
      nall++;
      if (fastOption_ && !trigger_fired) return false;
      if (fastOption_ && nmuonsForZ1>=1 && nmuonsForZ2>=2) return false;
      if (fastOption_ && njets>nJetMax_) return false;

      // Histograms per event shouldbe done only once, so keep track of them
      bool hlt_hist_done = false;
      bool met_hist_done = false;
      bool nz1_hist_done = false;
      bool nz2_hist_done = false;
      bool njets_hist_done = false;

      // Central W->mu nu selection criteria
      const int NFLAGS = 13;
      bool muon_sel[NFLAGS];
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
            //reco::TrackRef tk = mu.innerTrack();

            // Pt,eta cuts
            double pt = mu.pt();
            double eta = mu.eta();
            LogTrace("") << "\t... pt, eta: " << pt << " [GeV], " << eta;;
            if (pt>ptCut_) muon_sel[0] = true;
            else if (fastOption_) continue;
            if (fabs(eta)<etaCut_) muon_sel[1] = true;
            else if (fastOption_) continue;

            // d0, chi2, nhits quality cuts
            double dxy = gm->dxy(beamSpotHandle->position());
            double normalizedChi2 = gm->normalizedChi2();
            double validmuonhits=gm->hitPattern().numberOfValidMuonHits();
            //double standalonehits=mu.outerTrack()->numberOfValidHits();
            double trackerHits = gm->hitPattern().numberOfValidTrackerHits();
            LogTrace("") << "\t... dxy, normalizedChi2, trackerHits, isTrackerMuon?: " << dxy << " [cm], " << normalizedChi2 << ", " << trackerHits << ", " << mu.isTrackerMuon();
            if (fabs(dxy)<dxyCut_) muon_sel[2] = true;
            else if (fastOption_) continue;
            if (muon::isGoodMuon(mu,muon::GlobalMuonPromptTight)) muon_sel[3] = true;
            else if (fastOption_) continue;
            if (trackerHits>=trackerHitsCut_) muon_sel[4] = true;
            else if (fastOption_) continue;
            if (mu.isTrackerMuon()) muon_sel[5] = true;
            else if (fastOption_) continue;

            fill_histogram("PT_BEFORECUTS",pt);
            fill_histogram("ETA_BEFORECUTS",eta);
            fill_histogram("DXY_BEFORECUTS",dxy);
            fill_histogram("CHI2_BEFORECUTS",normalizedChi2);
            fill_histogram("NHITS_BEFORECUTS",trackerHits);
            fill_histogram("ValidMuonHits_BEFORECUTS",validmuonhits);
            fill_histogram("TKMU_BEFORECUTS",mu.isTrackerMuon());

            // Isolation cuts
            double isovar = mu.isolationR03().sumPt;
            if (isCombinedIso_) {
                  isovar += mu.isolationR03().emEt;
                  isovar += mu.isolationR03().hadEt;
            }
            if (isRelativeIso_) isovar /= pt;
            if (isovar<isoCut03_) muon_sel[6] = true;
            else if (fastOption_) continue;
            LogTrace("") << "\t... isolation value" << isovar <<", isolated? " << muon_sel[6];
            fill_histogram("ISO_BEFORECUTS",isovar);

            // HLT (not mtched to muon for the time being)
            if (trigger_fired) muon_sel[7] = true;
            else if (fastOption_) continue;

            // MET/MT cuts
            double w_et = met_et+ mu.pt();
            double w_px = met_px+ mu.px();
            double w_py = met_py+mu.py();
            double massT = w_et*w_et - w_px*w_px - w_py*w_py;
            massT = (massT>0) ? sqrt(massT) : 0;

            LogTrace("") << "\t... W mass, W_et, W_px, W_py: " << massT << ", " << w_et << ", " << w_px << ", " << w_py << " [GeV]";
            if (massT>mtMin_ && massT<mtMax_) muon_sel[8] = true;
            else if (fastOption_) continue;
            fill_histogram("MT_BEFORECUTS",massT);
            if (met_et>metMin_ && met_et<metMax_) muon_sel[9] = true;
            else if (fastOption_) continue;

            // Acoplanarity cuts
            Geom::Phi<double> deltaphi(mu.phi()-atan2(met_py,met_px));
            double acop = deltaphi.value();
            if (acop<0) acop = - acop;
            acop = M_PI - acop;
            LogTrace("") << "\t... acoplanarity: " << acop;
            if (acop<acopCut_) muon_sel[10] = true;
            else if (fastOption_) continue;
            fill_histogram("ACOP_BEFORECUTS",acop);

            // Remaining flags (from global event information)
            if (nmuonsForZ1<1 || nmuonsForZ2<2) muon_sel[11] = true;
            else if (fastOption_) continue;
            if (njets<=nJetMax_) muon_sel[12] = true;
            else if (fastOption_) continue;

            if (fastOption_) {
                  all_sel = true;
                  break;
            } else {
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
                        fill_histogram("PT_LASTCUT",pt);
                  if (!muon_sel[1] || flags_passed==NFLAGS)
                        fill_histogram("ETA_LASTCUT",eta);
                  if (!muon_sel[2] || flags_passed==NFLAGS)
                        fill_histogram("DXY_LASTCUT",dxy);
                  if (!muon_sel[3] || flags_passed==NFLAGS)
                        fill_histogram("CHI2_LASTCUT",normalizedChi2);
                        fill_histogram("ValidMuonHits_LASTCUT",validmuonhits);
                  if (!muon_sel[4] || flags_passed==NFLAGS)
                        fill_histogram("NHITS_LASTCUT",trackerHits);
                  if (!muon_sel[5] || flags_passed==NFLAGS)
                        fill_histogram("TKMU_LASTCUT",mu.isTrackerMuon());
                  if (!muon_sel[6] || flags_passed==NFLAGS)
                        fill_histogram("ISO_LASTCUT",isovar);
                  if (!muon_sel[7] || flags_passed==NFLAGS)
                        if (!hlt_hist_done) fill_histogram("TRIG_LASTCUT",trigger_fired);
                        hlt_hist_done = true;
                  if (!muon_sel[8] || flags_passed==NFLAGS)
                        fill_histogram("MT_LASTCUT",massT);
                  if (!muon_sel[9] || flags_passed==NFLAGS)
                        if (!met_hist_done) fill_histogram("MET_LASTCUT",met_et);
                        met_hist_done = true;
                  if (!muon_sel[10] || flags_passed==NFLAGS)
                        fill_histogram("ACOP_LASTCUT",acop);
                  if (!muon_sel[11] || flags_passed==NFLAGS)
                        if (!nz1_hist_done) fill_histogram("NZ1_LASTCUT",nmuonsForZ1);
                        nz1_hist_done = true;
                  if (!muon_sel[11] || flags_passed==NFLAGS)
                        if (!nz2_hist_done) fill_histogram("NZ2_LASTCUT",nmuonsForZ2);
                        nz2_hist_done = true;
                  if (!muon_sel[12] || flags_passed==NFLAGS)
                        if (!njets_hist_done) fill_histogram("NJETS_LASTCUT",njets);
                        njets_hist_done = true;
              }
            }

      }

      // Collect final flags
      if (!fastOption_) {
            if (rec_sel) nrec++;
            if (iso_sel) niso++;
            if (hlt_sel) nhlt++;
            if (met_sel) nmet++;
      }

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
