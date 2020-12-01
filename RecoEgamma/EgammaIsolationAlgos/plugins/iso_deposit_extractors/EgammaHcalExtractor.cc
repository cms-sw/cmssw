//*****************************************************************************
// File:      EgammaHcalExtractor.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "RecoCaloTools/Selectors/interface/CaloDualConeSelector.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaRecHitIsolation.h"

#include <Math/VectorUtil.h>

#include <vector>
#include <functional>

namespace egammaisolation {

  class EgammaHcalExtractor : public reco::isodeposit::IsoDepositExtractor {
  public:
    EgammaHcalExtractor(const edm::ParameterSet& par, edm::ConsumesCollector&& iC) : EgammaHcalExtractor(par, iC) {}
    EgammaHcalExtractor(const edm::ParameterSet& par, edm::ConsumesCollector& iC);

    ~EgammaHcalExtractor() override;

    void fillVetos(const edm::Event& ev, const edm::EventSetup& evSetup, const reco::TrackCollection& tracks) override {
    }
    reco::IsoDeposit deposit(const edm::Event& ev,
                             const edm::EventSetup& evSetup,
                             const reco::Track& track) const override {
      throw cms::Exception("Configuration Error")
          << "This extractor " << (typeid(this).name()) << " is not made for tracks";
    }
    reco::IsoDeposit deposit(const edm::Event& ev,
                             const edm::EventSetup& evSetup,
                             const reco::Candidate& c) const override;

  private:
    double extRadius_;
    double intRadius_;
    double etLow_;

    edm::EDGetTokenT<HBHERecHitCollection> hcalRecHitProducerToken_;
  };
}  // namespace egammaisolation

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaHcalExtractor, "EgammaHcalExtractor");

using namespace std;

using namespace egammaisolation;
using namespace reco::isodeposit;

EgammaHcalExtractor::EgammaHcalExtractor(const edm::ParameterSet& par, edm::ConsumesCollector& iC)
    : extRadius_(par.getParameter<double>("extRadius")),
      intRadius_(par.getParameter<double>("intRadius")),
      etLow_(par.getParameter<double>("etMin")),
      hcalRecHitProducerToken_(iC.consumes<HBHERecHitCollection>(par.getParameter<edm::InputTag>("hcalRecHits"))) {}

EgammaHcalExtractor::~EgammaHcalExtractor() {}

reco::IsoDeposit EgammaHcalExtractor::deposit(const edm::Event& iEvent,
                                              const edm::EventSetup& iSetup,
                                              const reco::Candidate& emObject) const {
  //Get MetaRecHit collection
  edm::Handle<HBHERecHitCollection> hcalRecHitHandle;
  iEvent.getByToken(hcalRecHitProducerToken_, hcalRecHitHandle);

  //Get Calo Geometry
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* caloGeom = pG.product();
  CaloDualConeSelector<HBHERecHit> coneSel(intRadius_, extRadius_, caloGeom, DetId::Hcal);

  //Take the SC position
  reco::SuperClusterRef sc = emObject.get<reco::SuperClusterRef>();
  math::XYZPoint caloPosition = sc->position();
  GlobalPoint point(caloPosition.x(), caloPosition.y(), caloPosition.z());
  // needed: coneSel.select(eta,phi,hits) is not the same!

  Direction candDir(caloPosition.eta(), caloPosition.phi());
  reco::IsoDeposit deposit(candDir);
  deposit.setVeto(reco::IsoDeposit::Veto(candDir, intRadius_));
  double sinTheta = sin(2 * atan(exp(-sc->eta())));
  deposit.addCandEnergy(sc->energy() * sinTheta);

  //Compute the HCAL energy behind ECAL
  coneSel.selectCallback(point, *hcalRecHitHandle, [&](const HBHERecHit& i) {
    const GlobalPoint& hcalHit_position = caloGeom->getPosition(i.detid());
    double hcalHit_eta = hcalHit_position.eta();
    double hcalHit_Et = i.energy() * sin(2 * atan(exp(-hcalHit_eta)));
    if (hcalHit_Et > etLow_) {
      deposit.addDeposit(Direction(hcalHit_eta, hcalHit_position.phi()), hcalHit_Et);
    }
  });

  return deposit;
}
