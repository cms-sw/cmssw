/* \class WMuNuSelector
 *
 * \author Juan Alcaraz, CIEMAT
 *
 */
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class WMuNuSelector : public edm::EDFilter {
public:
  WMuNuSelector (const edm::ParameterSet &);
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
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/GeometryVector/interface/Phi.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "DataFormats/Common/interface/View.h"
  
using namespace edm;
using namespace std;
using namespace reco;

WMuNuSelector::WMuNuSelector( const ParameterSet & cfg ) :
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

void WMuNuSelector::beginJob(const EventSetup &) {
      nall = 0;
      nrec = 0;
      niso = 0;
      nhlt = 0;
      nmet = 0;
      nsel = 0;
}

void WMuNuSelector::endJob() {
      double all = nall;
      double erec = nrec/all;
      double eiso = niso/all;
      double ehlt = nhlt/all;
      double emet = nmet/all;
      double esel = nsel/all;
      LogVerbatim("") << "\n>>>>>> W SELECTION SUMMARY BEGIN >>>>>>>>>>>>>>>";
      LogVerbatim("") << "Total numer of events analyzed: " << nall << " [events]";
      LogVerbatim("") << "Passing Pt/Eta/Quality cuts:    " << nrec << " [events], (" << setprecision(4) << erec*100. <<" +/- "<< setprecision(2) << sqrt(erec*(1-erec)/all)*100. << ")%";
      LogVerbatim("") << "Passing isolation cuts:         " << niso << " [events], (" << setprecision(4) << eiso*100. <<" +/- "<< setprecision(2) << sqrt(eiso*(1-eiso)/all)*100. << ")%, to previous step: " <<  setprecision(4) << eiso/erec*100 <<"%";
      LogVerbatim("") << "Passing HLT criteria:           " << nhlt << " [events], (" << setprecision(4) << ehlt*100. <<" +/- "<< setprecision(2) << sqrt(ehlt*(1-ehlt)/all)*100. << ")%, to previous step: " <<  setprecision(4) << ehlt/eiso*100 <<"%";
      LogVerbatim("") << "Passing MET/acoplanarity cuts:  " << nmet << " [events], (" << setprecision(4) << emet*100. <<" +/- "<< setprecision(2) << sqrt(emet*(1-emet)/all)*100. << ")%, to previous step: " <<  setprecision(4) << emet/ehlt*100 <<"%";
      LogVerbatim("") << "Passing Z/top rejection cuts:   " << nsel << " [events], (" << setprecision(4) << esel*100. <<" +/- "<< setprecision(2) << sqrt(esel*(1-esel)/all)*100. << ")%, to previous step: " <<  setprecision(4) << esel/emet*100 <<"%";
      LogVerbatim("") << ">>>>>> W SELECTION SUMMARY END   >>>>>>>>>>>>>>>\n";
}

bool WMuNuSelector::filter (Event & ev, const EventSetup &) {

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
      Handle <View<MET> > metCollection;
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

      // Jet collection
      Handle <View<Jet> > jetCollection;
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
      bool fired = false;
      int itrig1 = trigNames.triggerIndex(muonTrig_);
      if (triggerResults->accept(itrig1)) fired = true;
      LogTrace("") << ">>> Trigger bit: " << fired << " (" << muonTrig_ << ")";

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

      // Central W->mu nu selection criteria
      unsigned int nmuonsForW = 0;
      for (unsigned int i=0; i<muonCollectionSize; i++) {
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
            if (pt<ptCut_) continue;
            if (fabs(eta)>etaCut_) continue;

            // d0, chi2, nhits quality cuts
            double dxy = tk->dxy(beamSpotHandle->position());
            double normalizedChi2 = gm->normalizedChi2();
            double trackerHits = tk->numberOfValidHits();
            LogTrace("") << "\t... dxy, normalizedChi2, trackerHits, isTrackerMuon?: " << dxy << " [cm], " << normalizedChi2 << ", " << trackerHits << ", " << mu.isTrackerMuon();
            if (fabs(dxy)>dxyCut_) continue;
            if (normalizedChi2>normalizedChi2Cut_) continue;
            if (trackerHits<trackerHitsCut_) continue; 
            if (isAlsoTrackerMuon_ && !mu.isTrackerMuon()) continue;

            // "rec" => pt,eta and wuality cuts are satisfied
            nrec++;

            // Isolation cuts
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
            LogTrace("") << "\t... isolated? " << iso;

            // "iso" => "rec" AND "muon is isolated"
            if (!iso) continue;
            niso++;

            // "hlt" => "rec" AND "iso" AND "event is triggered"
            if (!fired) continue;
            nhlt++;

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
            if (met_et<metMin_) continue;
            if (met_et>metMax_) continue;
            if (massT<mtMin_) continue;
            if (massT>mtMax_) continue;

            // Acoplanarity cuts
            Geom::Phi<double> deltaphi(mu.phi()-atan2(met_py,met_px));
            double acop = deltaphi.value();
            if (acop<0) acop = - acop;
            acop = M_PI - acop;
            LogTrace("") << "\t... acoplanarity: " << acop;
            if (acop>acopCut_) continue;

            // "mt" => "rec" AND "iso" AND "hlt" AND "passes MET/MT and acoplanarity cuts"
            nmet++;

            nmuonsForW++;
      }

      // Z and top rejection is done here
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ1_ << " [GeV]: " << nmuonsForZ1;
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ2_ << " [GeV]: " << nmuonsForZ2;
      if (nmuonsForZ1>=1 && nmuonsForZ2>=2) {
            LogTrace("") << ">>>> Event REJECTED";
            return false;
      }
      LogTrace("") << ">>> Number of jets above " << eJetMin_ << " [GeV]: " << njets;
      if (njets>nJetMax_) return false;

      LogTrace("") << "> Number of muons for W: " << nmuonsForW;
      if (nmuonsForW<1) {
            LogTrace("") << ">>>> Event REJECTED";
            return false;
      }
      LogTrace("") << ">>>> Event SELECTED!!!";

      // "sel" => "rec" AND "iso" AND "hlt" AND "mt" AND "Z and top rejection"
      nsel++;

      return true;

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( WMuNuSelector );
