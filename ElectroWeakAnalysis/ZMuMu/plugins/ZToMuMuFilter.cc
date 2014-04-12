/* \class ZToMuMuFilter
 *
 * \author Juan Alcaraz, CIEMAT
 *
 */
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"

class ZToMuMuFilter : public edm::EDFilter {
public:
  ZToMuMuFilter(const edm::ParameterSet &);
private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  edm::EDGetTokenT<reco::CandidateCollection>  zCandsToken_;
  edm::EDGetTokenT<reco::CandDoubleAssociations> muIso1Token_;
  edm::EDGetTokenT<reco::CandDoubleAssociations> muIso2Token_;
  double ptMin_, etaMin_, etaMax_, massMin_, massMax_, isoMax_;
};

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
using namespace edm;
using namespace std;
using namespace reco;

ZToMuMuFilter::ZToMuMuFilter( const ParameterSet & cfg ) :
  zCandsToken_(consumes<CandidateCollection>(cfg.getParameter<InputTag>("zCands"))),
  muIso1Token_(consumes<CandDoubleAssociations>(cfg.getParameter<InputTag>("muonIsolations1"))),
  muIso2Token_(consumes<CandDoubleAssociations>(cfg.getParameter<InputTag>("muonIsolations2"))),
  ptMin_(cfg.getParameter<double>("ptMin")),
  etaMin_(cfg.getParameter<double>("etaMin")),
  etaMax_(cfg.getParameter<double>("etaMax")),
  massMin_(cfg.getParameter<double>("massMin")),
  massMax_(cfg.getParameter<double>("massMax")),
  isoMax_(cfg.getParameter<double>("isoMax")) {
}

bool ZToMuMuFilter::filter (Event & ev, const EventSetup &) {
  Handle<CandidateCollection> zCands;
  ev.getByToken(zCandsToken_, zCands);
  Handle<CandDoubleAssociations> muIso1;
  ev.getByToken(muIso1Token_, muIso1);
  Handle<CandDoubleAssociations> muIso2;
  ev.getByToken(muIso2Token_, muIso2);
  unsigned int nZ = zCands->size();
  if (nZ == 0) return false;
  for(unsigned int i = 0; i < nZ; ++ i) {
    const Candidate & zCand = (*zCands)[i];
    double zMass = zCand.mass();
    if (zMass < massMin_ || zMass > massMax_) return false;
    if(zCand.numberOfDaughters()!=2) return false;
    const Candidate * dau0 = zCand.daughter(0);
    const Candidate * dau1 = zCand.daughter(1);
    double pt0 = dau0->pt(), pt1 = dau1->pt();
    if (pt0 < ptMin_ || pt1 < ptMin_) return false;
    double eta0 = dau0->eta(), eta1 = dau1->eta();
    if(eta0 < etaMin_ || eta0 > etaMax_) return false;
    if(eta1 < etaMin_ || eta1 > etaMax_) return false;
    CandidateRef mu0 = dau0->masterClone().castTo<CandidateRef>();
    CandidateRef mu1 = dau1->masterClone().castTo<CandidateRef>();
    double iso0 = (*muIso1)[mu0];
    double iso1 = (*muIso2)[mu1];
    if (iso0 > isoMax_) return false;
    if (iso1 > isoMax_) return false;
  }
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ZToMuMuFilter );
