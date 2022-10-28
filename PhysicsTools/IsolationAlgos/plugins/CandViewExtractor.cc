#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"

#include <string>
#include <vector>

namespace muonisolation {

  class CandViewExtractor : public reco::isodeposit::IsoDepositExtractor {
  public:
    CandViewExtractor(){};
    CandViewExtractor(const edm::ParameterSet& par, edm::ConsumesCollector&& iC);

    ~CandViewExtractor() override {}

    void fillVetos(const edm::Event& ev, const edm::EventSetup& evSetup, const reco::TrackCollection& cand) override {}

    /*  virtual reco::IsoDeposit::Vetos vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Candidate & cand) const;

  virtual reco::IsoDeposit::Vetos vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Track & cand) const;
*/

    void initEvent(const edm::Event& ev, const edm::EventSetup& evSetup) override;

    reco::IsoDeposit deposit(const edm::Event& ev,
                             const edm::EventSetup& evSetup,
                             const reco::Track& muon) const override {
      return depositFromObject(ev, evSetup, muon);
    }

    reco::IsoDeposit deposit(const edm::Event& ev,
                             const edm::EventSetup& evSetup,
                             const reco::Candidate& muon) const override {
      return depositFromObject(ev, evSetup, muon);
    }

  private:
    reco::IsoDeposit::Veto veto(const reco::IsoDeposit::Direction& dir) const;

    template <typename T>
    reco::IsoDeposit depositFromObject(const edm::Event& ev, const edm::EventSetup& evSetup, const T& cand) const;

    // Parameter set
    edm::EDGetTokenT<edm::View<reco::Candidate> > theCandViewToken;  // Track Collection Label
    std::string theDepositLabel;                                     // name for deposit
    edm::Handle<edm::View<reco::Candidate> > theCandViewH;           //cached handle
    edm::Event::CacheIdentifier_t theCacheID;                        //event cacheID
    double theDiff_r;                                                // transverse distance to vertex
    double theDiff_z;                                                // z distance to vertex
    double theDR_Max;                                                // Maximum cone angle for deposits
    double theDR_Veto;                                               // Veto cone angle
  };

}  // namespace muonisolation

using namespace edm;
using namespace reco;
using namespace muonisolation;

template <typename T>
IsoDeposit CandViewExtractor::depositFromObject(const Event& event, const EventSetup& eventSetup, const T& cand) const {
  static const std::string metname = "MuonIsolation|CandViewExtractor";

  reco::isodeposit::Direction candDir(cand.eta(), cand.phi());
  IsoDeposit deposit(candDir);
  deposit.setVeto(veto(candDir));
  deposit.addCandEnergy(cand.pt());

  Handle<View<Candidate> > candViewH;
  if (theCacheID != event.cacheIdentifier()) {
    event.getByToken(theCandViewToken, candViewH);
  } else {
    candViewH = theCandViewH;
  }

  double eta = cand.eta(), phi = cand.phi();
  const reco::Particle::Point& vtx = cand.vertex();
  LogDebug(metname) << "cand eta=" << eta << " phi=" << phi << " vtx=" << vtx;
  for (View<Candidate>::const_iterator it = candViewH->begin(), ed = candViewH->end(); it != ed; ++it) {
    double dR = deltaR(it->eta(), it->phi(), eta, phi);
    LogDebug(metname) << "pdgid=" << it->pdgId() << " vtx=" << it->vertex() << " dR=" << dR
                      << " dvz=" << it->vz() - cand.vz() << " drho=" << (it->vertex() - vtx).Rho();
    if ((dR < theDR_Max) && (dR > theDR_Veto) && (std::abs(it->vz() - cand.vz()) < theDiff_z) &&
        ((it->vertex() - vtx).Rho() < theDiff_r)) {
      // ok
      reco::isodeposit::Direction dirTrk(it->eta(), it->phi());
      deposit.addDeposit(dirTrk, it->pt());
      LogDebug(metname) << "pt=" << it->pt();
    }
  }

  return deposit;
}

CandViewExtractor::CandViewExtractor(const ParameterSet& par, edm::ConsumesCollector&& iC)
    : theCandViewToken(iC.consumes<View<Candidate> >(par.getParameter<edm::InputTag>("inputCandView"))),
      theDepositLabel(par.getUntrackedParameter<std::string>("DepositLabel")),
      theDiff_r(par.getParameter<double>("Diff_r")),
      theDiff_z(par.getParameter<double>("Diff_z")),
      theDR_Max(par.getParameter<double>("DR_Max")),
      theDR_Veto(par.getParameter<double>("DR_Veto")) {}
/*
reco::IsoDeposit::Vetos CandViewExtractor::vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Candidate & cand) const
{
  reco::isodeposit::Direction dir(cand.eta(),cand.phi());
  return reco::IsoDeposit::Vetos(1,veto(dir));
}
*/

reco::IsoDeposit::Veto CandViewExtractor::veto(const reco::IsoDeposit::Direction& dir) const {
  reco::IsoDeposit::Veto result;
  result.vetoDir = dir;
  result.dR = theDR_Veto;
  return result;
}

void CandViewExtractor::initEvent(const edm::Event& ev, const edm::EventSetup& evSetup) {
  ev.getByToken(theCandViewToken, theCandViewH);
  theCacheID = ev.cacheIdentifier();
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, muonisolation::CandViewExtractor, "CandViewExtractor");
