#ifndef RecoEgamma_EgammaPhotonProducers_PhotonProducer_h
#define RecoEgamma_EgammaPhotonProducers_PhotonProducer_h
/** \class PhotonProducer
 **  
 **
 **  $Id: PhotonProducer.h,v 1.46 2013/02/27 20:33:00 eulisse Exp $ 
 **  $Date: 2013/02/27 20:33:00 $ 
 **  $Revision: 1.46 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
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
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonMIPHaloTagger.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h" 
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonEnergyCorrector.h"

// PhotonProducer inherits from EDProducer, so it can be a module:
class PhotonProducer : public edm::EDProducer {

 public:

  PhotonProducer (const edm::ParameterSet& ps);
  ~PhotonProducer();

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
			    const edm::Handle<CaloTowerCollection> & hcalTowersHandle,
			    //math::XYZPoint & vtx,
			    reco::VertexCollection& pvVertices,
			    reco::PhotonCollection & outputCollection,
			    int& iSC,
			    const EcalSeverityLevelAlgo * sevLv);

  // std::string PhotonCoreCollection_;
  std::string PhotonCollection_;
  edm::InputTag photonCoreProducer_;
  edm::InputTag barrelEcalHits_;
  edm::InputTag endcapEcalHits_;

  edm::InputTag hcalTowers_;

  std::string conversionProducer_;
  std::string conversionCollection_;

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
  std::string pixelSeedProducer_;
  std::string vertexProducer_;
  bool usePrimaryVertex_;
  edm::ParameterSet conf_;

  PositionCalc posCalculator_;

  edm::ESHandle<CaloGeometry> theCaloGeom_;
  edm::ESHandle<CaloTopology> theCaloTopo_;
 
  bool validPixelSeeds_;
  PhotonIsolationCalculator* thePhotonIsolationCalculator_;

  //MIP
  PhotonMIPHaloTagger* thePhotonMIPHaloTagger_;

  std::vector<double>  preselCutValuesBarrel_; 
  std::vector<double>  preselCutValuesEndcap_; 

  EcalClusterFunctionBaseClass* energyCorrectionF;
  PhotonEnergyCorrector* thePhotonEnergyCorrector_;
  std::string  candidateP4type_;

};
#endif
