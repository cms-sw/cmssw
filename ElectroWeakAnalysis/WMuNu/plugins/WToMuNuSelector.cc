/* \class WToMuNuSelector
 *
 * \author Juan Alcaraz, CIEMAT
 *
 */
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class WToMuNuSelector : public edm::EDFilter {
public:
  WToMuNuSelector (const edm::ParameterSet &);
  virtual bool filter(edm::Event&, const edm::EventSetup&);
private:
  edm::InputTag muonTag_;
  edm::InputTag metTag_;
  edm::InputTag isoTag_;
  edm::InputTag jetTag_;
  double ptThrForZCount_;
  double ptCut_;
  double etaCut_;
  double isoCone_;
  double isoCut_;
  double massTMin_;
  double massTMax_;
  double eJetMin_;
  int nJetMax_;
  double acopCut_;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/GeometryVector/interface/Phi.h"

using namespace edm;
using namespace std;
using namespace reco;

WToMuNuSelector::WToMuNuSelector( const ParameterSet & cfg ) :
      muonTag_(cfg.getParameter<edm::InputTag> ("MuonTag")),
      metTag_(cfg.getParameter<edm::InputTag> ("METTag")),
      isoTag_(cfg.getParameter<edm::InputTag> ("IsolationTag")),
      jetTag_(cfg.getParameter<edm::InputTag> ("JetTag")),
      ptThrForZCount_(cfg.getParameter<double>("PtThrForZCount")),
      ptCut_(cfg.getParameter<double>("PtCut")),
      etaCut_(cfg.getParameter<double>("EtaCut")),
      isoCone_(cfg.getParameter<double>("IsoCone")),
      isoCut_(cfg.getParameter<double>("IsoCut")),
      massTMin_(cfg.getParameter<double>("MassTMin")),
      massTMax_(cfg.getParameter<double>("MassTMax")),
      eJetMin_(cfg.getParameter<double>("EJetMin")),
      nJetMax_(cfg.getParameter<int>("NJetMax")),
      acopCut_(cfg.getParameter<double>("AcopCut"))
{
}

bool WToMuNuSelector::filter (Event & ev, const EventSetup &) {
      double met_px = 0.;
      double met_py = 0.;

      Handle<TrackCollection> muonCollection;
      ev.getByLabel(muonTag_, muonCollection);
      if (!muonCollection.isValid()) {
         LogTrace("") << ">>> Muon collection does not exist !!!";
         return false;
      }
      for (unsigned int i=0; i<muonCollection->size(); i++) {
         TrackRef mu(muonCollection,i);
         met_px -= mu->px();
         met_py -= mu->py();
      }

      Handle<CaloMETCollection> metCollection;
      ev.getByLabel(metTag_, metCollection);
      if (!metCollection.isValid()) {
         LogTrace("") << ">>> MET collection does not exist !!!";
         return false;
      }
      CaloMETCollection::const_iterator caloMET = metCollection->begin();
      LogTrace("") << ">>> CaloMET_et, CaloMET_py, CaloMET_py= " << caloMET->et() << ", " << caloMET->px() << ", " << caloMET->py();;
      met_px += caloMET->px();
      met_py += caloMET->py();
      double met_et = sqrt(met_px*met_px+met_py*met_py);

      edm::Handle<reco::MuIsoDepositAssociationMap> isodepMap;
      ev.getByLabel(isoTag_, isodepMap);
      if (!isodepMap.isValid()) {
         LogTrace("") << ">>> ISO collection does not exist !!!";
         return false;
      }

      Handle<CaloJetCollection> jetCollection;
      ev.getByLabel(jetTag_, jetCollection);
      if (!jetCollection.isValid()) {
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
      LogTrace("") << ">>> Number of muons: " << muonCollection->size();
      for (unsigned int i=0; i<muonCollection->size(); i++) {
          TrackRef mu(muonCollection,i);
          LogTrace("") << "> Processing muon number " << i << "...";
          double pt = mu->pt();
          if (pt>ptThrForZCount_) nmuonsForZ++;
          LogTrace("") << "\t... pt= " << pt << " GeV";
          if (pt<ptCut_) continue;
          double eta = mu->eta();
          LogTrace("") << "\t... eta= " << eta;
          if (fabs(eta)>etaCut_) continue;

          double w_et = mu->pt() + met_et;
          double w_px = mu->px() + met_px;
          double w_py = mu->py() + met_py;
          double massT = w_et*w_et - w_px*w_px - w_py*w_py;
          massT = (massT>0) ? sqrt(massT) : 0;
          LogTrace("") << "\t... W_et, W_px, W_py= " << w_et << ", " << w_px << ", " << w_py << " GeV";
          LogTrace("") << "\t... Invariant transverse mass= " << massT << " GeV";
          if (massT<massTMin_) continue;
          if (massT>massTMax_) continue;

          const reco::MuIsoDeposit dep = (*isodepMap)[mu];
          float ptsum = dep.depositWithin(isoCone_);
          LogTrace("") << "\t... Isol, Track pt= " << mu->pt() << " GeV, " << " ptsum = " << ptsum;
          if (ptsum >= isoCut_) continue;

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

DEFINE_FWK_MODULE( WToMuNuSelector );
