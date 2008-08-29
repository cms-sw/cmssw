/* \class WMuNuSelector
 *
 * \author Juan Alcaraz, CIEMAT
 *
 */
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class WMuNuSelector : public edm::EDFilter {
public:
  WMuNuSelector (const edm::ParameterSet &);
  virtual bool filter(edm::Event&, const edm::EventSetup&);
private:
  edm::InputTag muonTag_;
  edm::InputTag metTag_;
  edm::InputTag jetTag_;
  bool useOnlyGlobalMuons_;
  double ptCut_;
  double etaCut_;
  bool isRelativeIso_;
  double isoCut03_;
  double isoCut05_;
  double massTMin_;
  double massTMax_;
  double ptThrForZCount_;
  double acopCut_;
  double eJetMin_;
  int nJetMax_;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/GeometryVector/interface/Phi.h"

using namespace edm;
using namespace std;
using namespace reco;

WMuNuSelector::WMuNuSelector( const ParameterSet & cfg ) :
      muonTag_(cfg.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"))),
      metTag_(cfg.getUntrackedParameter<edm::InputTag> ("METTag", edm::InputTag("met"))),
      jetTag_(cfg.getUntrackedParameter<edm::InputTag> ("JetTag", edm::InputTag("iterativeCone5CaloJets"))),
      useOnlyGlobalMuons_(cfg.getUntrackedParameter<bool>("UseOnlyGlobalMuons", true)),
      ptCut_(cfg.getUntrackedParameter<double>("PtCut", 25.)),
      etaCut_(cfg.getUntrackedParameter<double>("EtaCut", 2.1)),
      isRelativeIso_(cfg.getUntrackedParameter<bool>("IsRelativeIso", true)),
      isoCut03_(cfg.getUntrackedParameter<double>("IsoCut03", 0.1)),
      isoCut05_(cfg.getUntrackedParameter<double>("IsoCut05", 999999.)),
      massTMin_(cfg.getUntrackedParameter<double>("MassTMin", 50.)),
      massTMax_(cfg.getUntrackedParameter<double>("MassTMax", 200.)),
      ptThrForZCount_(cfg.getUntrackedParameter<double>("PtThrForZCount", 20.)),
      acopCut_(cfg.getUntrackedParameter<double>("AcopCut", 999999.)),
      eJetMin_(cfg.getUntrackedParameter<double>("EJetMin", 999999.)),
      nJetMax_(cfg.getUntrackedParameter<int>("NJetMax", 999999))
{
}

bool WMuNuSelector::filter (Event & ev, const EventSetup &) {
      double met_px = 0.;
      double met_py = 0.;

      Handle<MuonCollection> muonCollection;
      if (!ev.getByLabel(muonTag_, muonCollection)) {
            LogTrace("") << ">>> Muon collection does not exist !!!";
            return false;
      }
      for (unsigned int i=0; i<muonCollection->size(); i++) {
            MuonRef mu(muonCollection,i);
            if (useOnlyGlobalMuons_ && !mu->isGlobalMuon()) continue;
            met_px -= mu->px();
            met_py -= mu->py();
      }

      Handle<CaloMETCollection> metCollection;
      if (!ev.getByLabel(metTag_, metCollection)) {
            LogTrace("") << ">>> MET collection does not exist !!!";
            return false;
      }
      CaloMETCollection::const_iterator caloMET = metCollection->begin();
      LogTrace("") << ">>> CaloMET_et, CaloMET_px, CaloMET_py= " << caloMET->et() << ", " << caloMET->px() << ", " << caloMET->py();;
      met_px += caloMET->px();
      met_py += caloMET->py();
      double met_et = sqrt(met_px*met_px+met_py*met_py);

      Handle<CaloJetCollection> jetCollection;
      if (!ev.getByLabel(jetTag_, jetCollection)) {
            LogTrace("") << ">>> CALOJET collection does not exist !!!";
            return false;
      }

      CaloJetCollection::const_iterator jet = jetCollection->begin();
      int njets = 0;
      for (jet=jetCollection->begin(); jet!=jetCollection->end(); jet++) {
            if (jet->et()>eJetMin_) njets++;
      }
      LogTrace("") << ">>> Number of jets " << jetCollection->size() << "; above " << eJetMin_ << " GeV: " << njets;
      LogTrace("") << ">>> Number of jets above " << eJetMin_ << " GeV: " << njets;
      if (njets>nJetMax_) return false;

      unsigned int nmuons = 0;
      unsigned int nmuonsForZ = 0;
      for (unsigned int i=0; i<muonCollection->size(); i++) {
            MuonRef mu(muonCollection,i);
            if (useOnlyGlobalMuons_ && !mu->isGlobalMuon()) continue;
            LogTrace("") << "> Processing (global) muon number " << i << "...";
            double pt = mu->pt();
            if (pt>ptThrForZCount_) nmuonsForZ++;
            LogTrace("") << "\t... pt= " << pt << " GeV";
            if (pt<ptCut_) continue;
            double eta = mu->eta();
            LogTrace("") << "\t... eta= " << eta;
            if (fabs(eta)>etaCut_) continue;
            double isovar03 = mu->isolationR03().sumPt/pt;
            LogTrace("") << "\t... iso sumPt deposit (in deltaR<0.3) / ptmuon= " << isovar03;
            if (isovar03>isoCut03_) continue;
            double isovar05 = mu->isolationR05().sumPt/pt;
            LogTrace("") << "\t... iso sumPt deposit (in deltaR<0.5) / ptmuon= " << isovar05;
            if (isovar05>isoCut05_) continue;
            double w_et = mu->pt() + met_et;
            double w_px = mu->px() + met_px;
            double w_py = mu->py() + met_py;
            double massT = w_et*w_et - w_px*w_px - w_py*w_py;
            massT = (massT>0) ? sqrt(massT) : 0;
            LogTrace("") << "\t... W_et, W_px, W_py= " << w_et << ", " << w_px << ", " << w_py << " GeV";
            LogTrace("") << "\t... Invariant transverse mass= " << massT << " GeV";
            if (massT<massTMin_) continue;
            if (massT>massTMax_) continue;
            Geom::Phi<double> deltaphi(mu->phi()-atan2(met_py,met_px));
            double acop = deltaphi.value();
            if (acop<0) acop = - acop;
            acop = M_PI - acop;
            LogTrace("") << "\t... acop= " << acop;
            if (acop>acopCut_) continue;
            nmuons++;
      }
      LogTrace("") << "> Muon counts to reject Z= " << nmuonsForZ;
      if (nmuonsForZ>=2) {
            LogTrace("") << ">>>> Event REJECTED";
            return false;
      }
      LogTrace("") << "> Number of muons for W= " << nmuons;
      if (nmuons<1) {
            LogTrace("") << ">>>> Event REJECTED";
            return false;
      }
      LogTrace("") << ">>>> Event SELECTED!!!";

      return true;

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( WMuNuSelector );
