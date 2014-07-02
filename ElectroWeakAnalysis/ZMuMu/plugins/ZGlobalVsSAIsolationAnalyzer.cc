#include "DataFormats/Common/interface/AssociationVector.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/PatCandidates/interface/Isolation.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include <vector>
#include <string>
#include <iostream>

using namespace edm;
using namespace std;
using namespace reco;
using namespace isodeposit;

class ZGlobalVsSAIsolationAnalyzer : public edm::EDAnalyzer {
public:
  typedef math::XYZVector Vector;
  ZGlobalVsSAIsolationAnalyzer(const edm::ParameterSet& cfg);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  virtual void endJob() override;
  EDGetTokenT<CandidateView> srcToken_;
  double dRVeto;
  double dRTrk, dREcal, dRHcal;
  double ptThreshold, etEcalThreshold, etHcalThreshold;
  double alpha, beta;
  double isoCut_;
  unsigned long selGlobal_, selSA_, totGlobal_, totSA_;
  bool isolated(const Direction & dir, const pat::IsoDeposit * trkIsoDep,
		const pat::IsoDeposit * ecalIsoDep, const pat::IsoDeposit * hcalIsoDep);
  void evaluate(const reco::Candidate* dau);
};

ZGlobalVsSAIsolationAnalyzer::ZGlobalVsSAIsolationAnalyzer(const ParameterSet& cfg):
  srcToken_(consumes<CandidateView>(cfg.getParameter<InputTag>("src"))),
  dRVeto(cfg.getParameter<double>("veto")),
  dRTrk(cfg.getParameter<double>("deltaRTrk")),
  dREcal(cfg.getParameter<double>("deltaREcal")),
  dRHcal(cfg.getParameter<double>("deltaRHcal")),
  ptThreshold(cfg.getParameter<double>("ptThreshold")),
  etEcalThreshold(cfg.getParameter<double>("etEcalThreshold")),
  etHcalThreshold(cfg.getParameter<double>("etHcalThreshold")),
  alpha(cfg.getParameter<double>("alpha")),
  beta(cfg.getParameter<double>("beta")),
  isoCut_(cfg.getParameter<double>("isoCut")),
  selGlobal_(0), selSA_(0), totGlobal_(0), totSA_(0) {
}

bool ZGlobalVsSAIsolationAnalyzer::isolated(const Direction & dir, const pat::IsoDeposit * trkIsoDep,
					    const pat::IsoDeposit * ecalIsoDep, const pat::IsoDeposit * hcalIsoDep) {
  IsoDeposit::AbsVetos vetoTrk, vetoEcal, vetoHcal;
  vetoTrk.push_back(new ConeVeto(dir, dRVeto));
  vetoTrk.push_back(new ThresholdVeto(ptThreshold));
  vetoEcal.push_back(new ConeVeto(dir, 0.));
  vetoEcal.push_back(new ThresholdVeto(etEcalThreshold));
  vetoHcal.push_back(new ConeVeto(dir, 0.));
  vetoHcal.push_back(new ThresholdVeto(etHcalThreshold));

  double trkIso = trkIsoDep->sumWithin(dir, dRTrk, vetoTrk);
  double ecalIso = ecalIsoDep->sumWithin(dir, dREcal, vetoEcal);
  double hcalIso = hcalIsoDep->sumWithin(dir, dRHcal, vetoHcal);
  double iso = alpha*((0.5*(1+beta)*ecalIso) + (0.5*(1-beta)*hcalIso)) + (1-alpha)*trkIso;
  return iso < isoCut_;
}

void ZGlobalVsSAIsolationAnalyzer::evaluate(const reco::Candidate* dau) {
  const pat::Muon * mu = dynamic_cast<const pat::Muon *>(&*dau->masterClone());
  if(mu == 0) throw Exception(errors::InvalidReference) << "Daughter is not a muon!\n";
  const pat::IsoDeposit * trkIsoDep = mu->isoDeposit(pat::TrackIso);
  const pat::IsoDeposit * ecalIsoDep = mu->isoDeposit(pat::EcalIso);
  const pat::IsoDeposit * hcalIsoDep = mu->isoDeposit(pat::HcalIso);
  // global muon
  {
    Direction dir = Direction(mu->eta(), mu->phi());
    if(isolated(dir, trkIsoDep, ecalIsoDep, hcalIsoDep)) selGlobal_++;
    totGlobal_++;
  }
  // stand-alone
  {
    TrackRef sa = dau->get<TrackRef,reco::StandAloneMuonTag>();
    Direction dir = Direction(sa->eta(), sa->phi());
    if(isolated(dir, trkIsoDep, ecalIsoDep, hcalIsoDep)) selSA_++;
    totSA_++;
  }
}

void ZGlobalVsSAIsolationAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  Handle<CandidateView> dimuons;
  event.getByToken(srcToken_, dimuons);
  for(unsigned int i=0; i< dimuons->size(); ++ i) {
    const Candidate & zmm = (* dimuons)[i];
    evaluate(zmm.daughter(0));
    evaluate(zmm.daughter(1));
  }
}

void ZGlobalVsSAIsolationAnalyzer::endJob() {
  cout << "Isolation efficiency report:" << endl;
  double eff, err;
  eff =  double(selGlobal_)/double(totGlobal_);
  err = sqrt(eff*(1.-eff)/double(totGlobal_));
  cout <<"Global: " << selGlobal_ << "/" << totGlobal_ << " = " <<eff <<"+/-" << err<< endl;
  eff = double(selSA_)/double(totSA_);
  err = sqrt(eff*(1.-eff)/double(totSA_));
  cout <<"St.Al.: " << selSA_ << "/" << totSA_ << " = " << eff <<"+/-" << err << endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZGlobalVsSAIsolationAnalyzer);
