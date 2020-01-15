#ifndef RecoEgamma_EgammaPhotonProducers_PhotonProducer_h
#define RecoEgamma_EgammaPhotonProducers_PhotonProducer_h
/** \class PhotonProducer
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonMIPHaloTagger.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonEnergyCorrector.h"

// PhotonProducer inherits from EDProducer, so it can be a module:
class PhotonProducer : public edm::stream::EDProducer<> {
public:
  PhotonProducer(const edm::ParameterSet& ps);

  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  void fillPhotonCollection(edm::Event& evt,
                            edm::EventSetup const& es,
                            const edm::Handle<reco::PhotonCoreCollection>& photonCoreHandle,
                            const CaloTopology* topology,
                            const EcalRecHitCollection* ecalBarrelHits,
                            const EcalRecHitCollection* ecalEndcapHits,
                            CaloTowerCollection const& hcalTowers,
                            reco::VertexCollection& pvVertices,
                            reco::PhotonCollection& outputCollection,
                            int& iSC,
                            const EcalSeverityLevelAlgo* sevLv);

  // std::string PhotonCoreCollection_;
  std::string PhotonCollection_;
  edm::EDGetTokenT<reco::PhotonCoreCollection> photonCoreProducer_;
  edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHits_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHits_;
  edm::EDGetTokenT<CaloTowerCollection> hcalTowers_;
  edm::EDGetTokenT<reco::VertexCollection> vertexProducer_;

  //AA
  //Flags and severities to be excluded from calculations

  std::vector<int> flagsexclEB_;
  std::vector<int> flagsexclEE_;
  std::vector<int> severitiesexclEB_;
  std::vector<int> severitiesexclEE_;

  double hOverEConeSize_;
  double maxHOverE_;
  double minSCEt_;
  double highEt_;
  double minR9Barrel_;
  double minR9Endcap_;
  bool runMIPTagger_;

  bool validConversions_;

  bool usePrimaryVertex_;

  PositionCalc posCalculator_;

  bool validPixelSeeds_;
  PhotonIsolationCalculator photonIsolationCalculator_;

  //MIP
  PhotonMIPHaloTagger photonMIPHaloTagger_;

  std::vector<double> preselCutValuesBarrel_;
  std::vector<double> preselCutValuesEndcap_;

  PhotonEnergyCorrector photonEnergyCorrector_;
  std::string candidateP4type_;
};
#endif
