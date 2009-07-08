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

  unsigned int nall;
  unsigned int nrec;
  unsigned int niso;
  unsigned int nhlt;
  unsigned int nmt;
  unsigned int ntop;
  unsigned int nsel;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
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
}

void WMuNuSelector::beginJob(const EventSetup &) {
      nall = 0;
      nrec = 0;
      niso = 0;
      nhlt = 0;
      nmt = 0;
      ntop = 0;
      nsel = 0;
}

void WMuNuSelector::endJob() {
      LogError("") << "nall= " << nall << endl;
      double all = nall;
      double erec = nrec/all;
      double eiso = niso/all;
      double ehlt = nhlt/all;
      double emt  = nmt /all;
      double etop = ntop/all;
      double esel = nsel/all;
      LogError("") << "nrec= " << nrec << ", " << erec*100 <<" +/- "<<sqrt(erec*(1-erec)/all)*100;
      LogError("") << "niso= " << niso << ", " << eiso*100 <<" +/- "<<sqrt(eiso*(1-eiso)/all)*100<<" %, to previous step: " << eiso/erec*100 <<"%";
      LogError("") << "nhlt= " << nhlt << ", " << ehlt*100 <<" +/- "<<sqrt(ehlt*(1-ehlt)/all)*100<< " %, to previous step: " << ehlt/eiso*100 <<"%";
      LogError("") << "nmt = " << nmt  << ", " << emt*100 <<" +/- "<<sqrt(emt*(1-emt)/all)*100<< "%, to previous step: " << emt/ehlt*100 <<"%";
      LogError("") << "ntop= " << ntop << ", " << etop*100 <<" +/- "<<sqrt(etop*(1-etop)/all)*100<<  "%, to previous step: " << etop/emt*100 <<"%";
      LogError("") << "nsel= " << nsel << ", " << esel*100 <<" +/- "<<sqrt(esel*(1-esel)/all)*100<< "%, to previous step: " << esel/etop*100 <<"%";
}

bool WMuNuSelector::filter (Event & ev, const EventSetup &) {

      // Muon collection
      Handle<View<Muon> > muonCollection;
      if (!ev.getByLabel(muonTag_, muonCollection)) {
            LogError("") << ">>> Muon collection does not exist !!!";
            return false;
      }
      unsigned int muonCollectionSize = muonCollection->size();

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
                  if (useOnlyGlobalMuons_ && !mu.isGlobalMuon()) continue;
                  met_px -= mu.px();
                  met_py -= mu.py();
            }
      }
      double met_et = sqrt(met_px*met_px+met_py*met_py);
      LogTrace("") << ">>> MET, MET_px, MET_py= " << met_et << ", " << met_px << ", " << met_py;

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
      LogTrace("") << ">>> Total number of jets= " << jetCollectionSize;

      // Trigger
      Handle<TriggerResults> triggerResults;
      TriggerNames trigNames;
      if (!ev.getByLabel(trigTag_, triggerResults)) {
            LogError("") << ">>> TRIGGER collection does not exist !!!";
            return false;
      }
      trigNames.init(*triggerResults);
      bool fired = false;

      /*
      for (unsigned int i=0; i<triggerResults->size(); i++) {
            if (triggerResults->accept(i)) {
                  LogError("") << "Accept by: " << i << ", Trigger: " << trigNames.triggerName(i);
            }
      }
      */

      int itrig1 = trigNames.triggerIndex(muonTrig_);
      if (triggerResults->accept(itrig1)) fired = true;

      nall++;
      unsigned int nmuons = 0;
      unsigned int nmuonsForZ1 = 0;
      unsigned int nmuonsForZ2 = 0;
      for (unsigned int i=0; i<muonCollectionSize; i++) {
            const Muon& mu = muonCollection->at(i);
            if (useOnlyGlobalMuons_ && !mu.isGlobalMuon()) continue;
            if (mu.innerTrack().isNull()) continue;
            TrackRef tk = mu.innerTrack();
            LogTrace("") << "> Processing muon number " << i << "...";

            double pt = tk->pt();
            if (pt>ptThrForZ1_) nmuonsForZ1++;
            if (pt>ptThrForZ2_) nmuonsForZ2++;
            LogTrace("") << "\t... pt= " << pt << " GeV";

            double eta = tk->eta();
            LogTrace("") << "\t... eta= " << eta;

            if (pt<ptCut_) continue;
            if (fabs(eta)>etaCut_) continue;
            nrec++;

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
            if (!iso) continue;
            niso++;

            if (!fired) continue;
            nhlt++;

            double w_et = tk->pt() + met_et;
            double w_px = tk->px() + met_px;
            double w_py = tk->py() + met_py;
            double massT = w_et*w_et - w_px*w_px - w_py*w_py;
            massT = (massT>0) ? sqrt(massT) : 0;
            LogTrace("") << "\t... W_et, W_px, W_py= " << w_et << ", " << w_px << ", " << w_py << " GeV";
            LogTrace("") << "\t... Invariant transverse mass= " << massT << " GeV";
            if (massT<massTMin_) continue;
            if (massT>massTMax_) continue;
            nmt++;

            LogTrace("") << ">>> Number of jets above " << eJetMin_ << " GeV: " << njets;
            if (njets>nJetMax_) return false;

            Geom::Phi<double> deltaphi(tk->phi()-atan2(met_py,met_px));
            double acop = deltaphi.value();
            if (acop<0) acop = - acop;
            acop = M_PI - acop;
            LogTrace("") << "\t... acop= " << acop;
            if (acop>acopCut_) continue;
            ntop++;
            nmuons++;
      }
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ1_ << " GeV = " << nmuonsForZ1;
      LogTrace("") << "> Z rejection: muons above " << ptThrForZ2_ << " GeV = " << nmuonsForZ2;
      if (nmuonsForZ1>=1 && nmuonsForZ2>=2) {
            LogTrace("") << ">>>> Event REJECTED";
            return false;
      }
      LogTrace("") << "> Number of muons for W= " << nmuons;
      if (nmuons<1) {
            LogTrace("") << ">>>> Event REJECTED";
            return false;
      }
      LogTrace("") << ">>>> Event SELECTED!!!";
      nsel++;

      return true;

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( WMuNuSelector );
