// -*- C++ -*-
//
// Package:    EgammaIsoDetIdCollectionProducer
// Class:      EgammaIsoDetIdCollectionProducer
//
/**\class EgammaIsoDetIdCollectionProducer 
Original author: Matthew LeBourgeois PH/CMG
Modified from :
RecoEcal/EgammaClusterProducers/{src,interface}/InterestingDetIdCollectionProducer.{h,cc}
by Paolo Meridiani PH/CMG
 
Implementation:
 <Notes on implementation>
*/

#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoCaloTools/Selectors/interface/CaloDualConeSelector.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

template <class T1>
class EgammaIsoDetIdCollectionProducer : public edm::global::EDProducer<> {
public:
  typedef std::vector<T1> T1Collection;
  //! ctor
  explicit EgammaIsoDetIdCollectionProducer(const edm::ParameterSet&);
  //! producer
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------
  edm::EDGetTokenT<EcalRecHitCollection> recHitsToken_;
  edm::EDGetTokenT<T1Collection> emObjectToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> sevLvToken_;
  edm::InputTag recHitsLabel_;
  edm::InputTag emObjectLabel_;
  double energyCut_;
  double etCut_;
  double etCandCut_;
  double outerRadius_;
  double innerRadius_;
  std::string interestingDetIdCollection_;

  std::vector<int> severitiesexclEB_;
  std::vector<int> severitiesexclEE_;
  std::vector<int> flagsexclEB_;
  std::vector<int> flagsexclEE_;
};

template <class T1>
EgammaIsoDetIdCollectionProducer<T1>::EgammaIsoDetIdCollectionProducer(const edm::ParameterSet& iConfig)
    : recHitsToken_{consumes(iConfig.getParameter<edm::InputTag>("recHitsLabel"))},
      emObjectToken_{consumes(iConfig.getParameter<edm::InputTag>("emObjectLabel"))},
      caloGeometryToken_(esConsumes()),
      sevLvToken_(esConsumes()),
      recHitsLabel_(iConfig.getParameter<edm::InputTag>("recHitsLabel")),
      emObjectLabel_(iConfig.getParameter<edm::InputTag>("emObjectLabel")),
      energyCut_(iConfig.getParameter<double>("energyCut")),
      etCut_(iConfig.getParameter<double>("etCut")),
      etCandCut_(iConfig.getParameter<double>("etCandCut")),
      outerRadius_(iConfig.getParameter<double>("outerRadius")),
      innerRadius_(iConfig.getParameter<double>("innerRadius")),
      interestingDetIdCollection_(iConfig.getParameter<std::string>("interestingDetIdCollection")) {
  auto const& flagnamesEB = iConfig.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEB");
  auto const& flagnamesEE = iConfig.getParameter<std::vector<std::string>>("RecHitFlagToBeExcludedEE");

  flagsexclEB_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEB);
  flagsexclEE_ = StringToEnumValue<EcalRecHit::Flags>(flagnamesEE);

  auto const& severitynamesEB = iConfig.getParameter<std::vector<std::string>>("RecHitSeverityToBeExcludedEB");

  severitiesexclEB_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEB);

  auto const& severitynamesEE = iConfig.getParameter<std::vector<std::string>>("RecHitSeverityToBeExcludedEE");

  severitiesexclEE_ = StringToEnumValue<EcalSeverityLevel::SeverityLevel>(severitynamesEE);

  //register your products
  produces<DetIdCollection>(interestingDetIdCollection_);
}

// ------------ method called to produce the data  ------------
template <class T1>
void EgammaIsoDetIdCollectionProducer<T1>::produce(edm::StreamID,
                                                   edm::Event& iEvent,
                                                   const edm::EventSetup& iSetup) const {
  using namespace edm;
  using namespace std;

  auto const& emObjects = iEvent.get(emObjectToken_);
  auto const& ecalRecHits = iEvent.get(recHitsToken_);

  edm::ESHandle<CaloGeometry> pG = iSetup.getHandle(caloGeometryToken_);
  const CaloGeometry* caloGeom = pG.product();

  edm::ESHandle<EcalSeverityLevelAlgo> sevlv = iSetup.getHandle(sevLvToken_);
  const EcalSeverityLevelAlgo* sevLevel = sevlv.product();

  std::unique_ptr<CaloDualConeSelector<EcalRecHit>> doubleConeSel_ = nullptr;
  if (recHitsLabel_.instance() == "EcalRecHitsEB") {
    doubleConeSel_ =
        std::make_unique<CaloDualConeSelector<EcalRecHit>>(innerRadius_, outerRadius_, &*pG, DetId::Ecal, EcalBarrel);
  } else if (recHitsLabel_.instance() == "EcalRecHitsEE") {
    doubleConeSel_ =
        std::make_unique<CaloDualConeSelector<EcalRecHit>>(innerRadius_, outerRadius_, &*pG, DetId::Ecal, EcalEndcap);
  }

  //Create empty output collections
  auto detIdCollection = std::make_unique<DetIdCollection>();

  if (doubleConeSel_) {                    //if cone selector was created
    for (auto const& emObj : emObjects) {  //Loop over candidates

      if (emObj.et() < etCandCut_)
        continue;  //don't calculate if object hasn't enough energy

      GlobalPoint pclu(emObj.caloPosition().x(), emObj.caloPosition().y(), emObj.caloPosition().z());
      doubleConeSel_->selectCallback(pclu, ecalRecHits, [&](const EcalRecHit& recIt) {
        if (recIt.energy() < energyCut_)
          return;  //dont fill if below E noise value

        double et =
            recIt.energy() * caloGeom->getPosition(recIt.detid()).perp() / caloGeom->getPosition(recIt.detid()).mag();

        if (et < etCut_)
          return;  //dont fill if below ET noise value

        bool isBarrel = false;
        if (fabs(caloGeom->getPosition(recIt.detid()).eta() < 1.479))
          isBarrel = true;

        int severityFlag = sevLevel->severityLevel(recIt.detid(), ecalRecHits);
        if (isBarrel) {
          auto sit = std::find(severitiesexclEB_.begin(), severitiesexclEB_.end(), severityFlag);
          if (sit != severitiesexclEB_.end())
            return;
        } else {
          auto sit = std::find(severitiesexclEE_.begin(), severitiesexclEE_.end(), severityFlag);
          if (sit != severitiesexclEE_.end())
            return;
        }

        if (isBarrel) {
          // new rechit flag checks
          if (!recIt.checkFlag(EcalRecHit::kGood)) {
            if (recIt.checkFlags(flagsexclEB_)) {
              return;
            }
          }
        } else {
          // new rechit flag checks
          if (!recIt.checkFlag(EcalRecHit::kGood)) {
            if (recIt.checkFlags(flagsexclEE_)) {
              return;
            }
          }
        }

        if (std::find(detIdCollection->begin(), detIdCollection->end(), recIt.detid()) == detIdCollection->end())
          detIdCollection->push_back(recIt.detid());
      });  //end rechits

    }  //end candidates

  }  //end if cone selector was created

  iEvent.put(std::move(detIdCollection), interestingDetIdCollection_);
}

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using EleIsoDetIdCollectionProducer = EgammaIsoDetIdCollectionProducer<reco::GsfElectron>;
using GamIsoDetIdCollectionProducer = EgammaIsoDetIdCollectionProducer<reco::Photon>;

DEFINE_FWK_MODULE(GamIsoDetIdCollectionProducer);
DEFINE_FWK_MODULE(EleIsoDetIdCollectionProducer);
