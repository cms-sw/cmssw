//*****************************************************************************
// File:      EgammaTowerExtractor.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

#include <Math/VectorUtil.h>

#include <vector>
#include <functional>
#include <cmath>

namespace egammaisolation {

  class EgammaTowerExtractor : public reco::isodeposit::IsoDepositExtractor {
  public:
    enum HcalDepth { AllDepths = -1, Undefined = 0, Depth1 = 1, Depth2 = 2 };

  public:
    EgammaTowerExtractor(const edm::ParameterSet& par, edm::ConsumesCollector&& iC) : EgammaTowerExtractor(par, iC) {}
    EgammaTowerExtractor(const edm::ParameterSet& par, edm::ConsumesCollector& iC)
        : extRadius2_(par.getParameter<double>("extRadius")),
          intRadius_(par.getParameter<double>("intRadius")),
          etLow_(par.getParameter<double>("etMin")),
          caloTowerToken(iC.consumes<CaloTowerCollection>(par.getParameter<edm::InputTag>("caloTowers"))),
          depth_(par.getParameter<int>("hcalDepth")) {
      extRadius2_ *= extRadius2_;
      //lets just check we have a valid depth
      //should we throw an exception or just warn and then fail gracefully later?
      if (depth_ != AllDepths && depth_ != Depth1 && depth_ != Depth2) {
        throw cms::Exception("Configuration Error")
            << "hcalDepth passed to EgammaTowerExtractor is invalid " << std::endl;
      }
    }

    ~EgammaTowerExtractor() override;

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
    double extRadius2_;
    double intRadius_;
    double etLow_;

    edm::EDGetTokenT<CaloTowerCollection> caloTowerToken;
    int depth_;
    //const CaloTowerCollection *towercollection_ ;
  };
}  // namespace egammaisolation

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaTowerExtractor, "EgammaTowerExtractor");

using namespace ROOT::Math::VectorUtil;

using namespace egammaisolation;
using namespace reco::isodeposit;

EgammaTowerExtractor::~EgammaTowerExtractor() {}

reco::IsoDeposit EgammaTowerExtractor::deposit(const edm::Event &iEvent,
                                               const edm::EventSetup &iSetup,
                                               const reco::Candidate &emObject) const {
  edm::Handle<CaloTowerCollection> towercollectionH;
  iEvent.getByToken(caloTowerToken, towercollectionH);

  //Take the SC position
  reco::SuperClusterRef sc = emObject.get<reco::SuperClusterRef>();
  math::XYZPoint caloPosition = sc->position();

  Direction candDir(caloPosition.eta(), caloPosition.phi());
  reco::IsoDeposit deposit(candDir);
  deposit.setVeto(reco::IsoDeposit::Veto(candDir, intRadius_));
  deposit.addCandEnergy(sc->energy() * sin(2 * atan(exp(-sc->eta()))));

  //loop over tracks
  for (CaloTowerCollection::const_iterator trItr = towercollectionH->begin(), trEnd = towercollectionH->end();
       trItr != trEnd;
       ++trItr) {
    double depEt = 0;
    //the hcal can be seperated into different depths
    //currently it is setup to check that the depth is valid in constructor
    //if the depth is not valid it fails gracefully
    //small bug fix, hadEnergyHeInnerLater returns zero for towers which are only depth 1
    //but we want Depth1 isolation to include these so we have to manually check for this
    if (depth_ == AllDepths)
      depEt = trItr->hadEt();
    else if (depth_ == Depth1)
      depEt = trItr->ietaAbs() < 18 || trItr->ietaAbs() > 29
                  ? trItr->hadEt()
                  : trItr->hadEnergyHeInnerLayer() * sin(trItr->p4().theta());
    else if (depth_ == Depth2)
      depEt = trItr->hadEnergyHeOuterLayer() * sin(trItr->p4().theta());

    if (depEt < etLow_)
      continue;

    Direction towerDir(trItr->eta(), trItr->phi());
    double dR2 = candDir.deltaR2(towerDir);

    if (dR2 < extRadius2_) {
      deposit.addDeposit(towerDir, depEt);
    }

  }  //end loop over tracks

  return deposit;
}
