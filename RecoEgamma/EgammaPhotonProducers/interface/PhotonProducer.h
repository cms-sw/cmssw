#ifndef RecoEgamma_EgammaPhotonProducers_PhotonProducer_h
#define RecoEgamma_EgammaPhotonProducers_PhotonProducer_h
/** \class PhotonProducer
 **  
 **
 **  $Id: PhotonProducer.h,v 1.33 2009/03/04 21:20:09 nancy Exp $ 
 **  $Date: 2009/03/04 21:20:09 $ 
 **  $Revision: 1.33 $
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
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "RecoEgamma/EgammaTools/interface/HoECalculator.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"

// PhotonProducer inherits from EDProducer, so it can be a module:
class PhotonProducer : public edm::EDProducer {

 public:

  PhotonProducer (const edm::ParameterSet& ps);
  ~PhotonProducer();

  virtual void beginRun (edm::Run& r, edm::EventSetup const & es);
  virtual void endRun(edm::Run &,  edm::EventSetup const&);
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  void fillPhotonCollection(edm::Event& evt,
			    edm::EventSetup const & es,
                            const edm::Handle<reco::PhotonCoreCollection> & photonCoreHandle,
                            const CaloTopology *topology,
			    const EcalRecHitCollection* ecalBarrelHits,
			    const EcalRecHitCollection* ecalEndcapHits,
			    const edm::Handle<CaloTowerCollection> & hcalTowersHandle,
			    math::XYZPoint & vtx,
			    reco::PhotonCollection & outputCollection,
			    int& iSC);



 

  std::string PhotonCoreCollection_;
  std::string PhotonCollection_;
  edm::InputTag photonCoreProducer_;
  edm::InputTag barrelEcalHits_;
  edm::InputTag endcapEcalHits_;

  edm::InputTag hcalTowers_;

  std::string conversionProducer_;
  std::string conversionCollection_;


  std::string hbheLabel_;
  std::string hbheInstanceName_;
  double hOverEConeSize_;
  double maxHOverE_;
  double minSCEt_;
  double highEt_;
  double  minR9Barrel_;
  double  minR9Endcap_;

  bool validConversions_;
  std::string pixelSeedProducer_;
  std::string vertexProducer_;
  bool usePrimaryVertex_;
  edm::ParameterSet conf_;

  PositionCalc posCalculator_;


  edm::ESHandle<CaloGeometry> theCaloGeom_;
  edm::ESHandle<CaloTopology> theCaloTopo_;
  HoECalculator  theHoverEcalc_;


  bool validPixelSeeds_;
  PhotonIsolationCalculator* thePhotonIsolationCalculator_;


  std::vector<double>  preselCutValuesBarrel_; 
  std::vector<double>  preselCutValuesEndcap_; 
  //int nEvt_;

};
#endif
