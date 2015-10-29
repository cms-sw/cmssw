#ifndef RecoEgamma_EgammaPhotonProducers_GEDPhotonProducer_h
#define RecoEgamma_EgammaPhotonProducers_GEDPhotonProducer_h
/** \class GEDPhotonProducer
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
#include "RecoEgamma/PhotonIdentification/interface/PFPhotonIsolationCalculator.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonMIPHaloTagger.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h" 
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonEnergyCorrector.h"

#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

// GEDPhotonProducer inherits from EDProducer, so it can be a module:
class GEDPhotonProducer : public edm::stream::EDProducer<> {

 public:

  GEDPhotonProducer (const edm::ParameterSet& ps);
  ~GEDPhotonProducer();

  virtual void beginRun (edm::Run const& r, edm::EventSetup const & es) override final;
  virtual void endRun(edm::Run const&,  edm::EventSetup const&) override final;
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  void fillPhotonCollection(edm::Event& evt,
			    edm::EventSetup const & es,
                            const edm::Handle<reco::PhotonCoreCollection> & photonCoreHandle,
                            const CaloTopology *topology,
			    const EcalRecHitCollection* ecalBarrelHits,
			    const EcalRecHitCollection* ecalEndcapHits,
                            const EcalRecHitCollection* preshowerHits,
			    const edm::Handle<CaloTowerCollection> & hcalTowersHandle,
			    reco::VertexCollection& pvVertices,
			    reco::PhotonCollection & outputCollection,
			    int& iSC);


 void fillPhotonCollection(edm::Event& evt,
			    edm::EventSetup const & es,
			   const edm::Handle<reco::PhotonCollection> & photonHandle,
		   	   const edm::Handle<reco::PFCandidateCollection> pfCandidateHandle,
			   const edm::Handle<reco::PFCandidateCollection> pfEGCandidateHandle,
			   edm::ValueMap<reco::PhotonRef>  pfEGCandToPhotonMap,
			   edm::Handle< reco::VertexCollection >&  pvVertices,
			   reco::PhotonCollection & outputCollection,
			   int& iSC);


 // std::string PhotonCoreCollection_;
 std::string photonCollection_;
 edm::InputTag  photonProducer_;
 
 edm::EDGetTokenT<reco::PhotonCoreCollection> photonCoreProducerT_;
 edm::EDGetTokenT<reco::PhotonCollection> photonProducerT_;
 edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHits_;
 edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHits_;
 edm::EDGetTokenT<EcalRecHitCollection> preshowerHits_;
 edm::EDGetTokenT<reco::PFCandidateCollection> pfEgammaCandidates_;
 edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidates_;
 edm::EDGetTokenT<CaloTowerCollection> hcalTowers_;
 edm::EDGetTokenT<reco::VertexCollection> vertexProducer_;
 
  std::string conversionProducer_;
  std::string conversionCollection_;
  std::string valueMapPFCandPhoton_;

  PFPhotonIsolationCalculator* thePFBasedIsolationCalculator_;
  PhotonIsolationCalculator* thePhotonIsolationCalculator_;

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
  double  minR9Barrel_;
  double  minR9Endcap_;
  bool   runMIPTagger_;

  bool validConversions_;
  std::string reconstructionStep_;

  bool usePrimaryVertex_;
  edm::ParameterSet conf_;
  PositionCalc posCalculator_;

  edm::ESHandle<CaloGeometry> theCaloGeom_;
  edm::ESHandle<CaloTopology> theCaloTopo_;
 
  bool validPixelSeeds_;

  //MIP
  PhotonMIPHaloTagger* thePhotonMIPHaloTagger_;

  std::vector<double>  preselCutValuesBarrel_; 
  std::vector<double>  preselCutValuesEndcap_; 

  EcalClusterFunctionBaseClass* energyCorrectionF;
  PhotonEnergyCorrector* thePhotonEnergyCorrector_;
  std::string  candidateP4type_;

};
#endif
