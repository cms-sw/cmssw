#ifndef RecoEgamma_EgammaPhotonProducers_PhotonCoreProducer_h
#define RecoEgamma_EgammaPhotonProducers_PhotonCoreProducer_h
/** \class PhotonCoreProducer
 **  
 **
 **  $Id: PhotonCoreProducer.h,v 1.4 2013/02/27 20:33:00 eulisse Exp $ 
 **  $Date: 2013/02/27 20:33:00 $ 
 **  $Revision: 1.4 $
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
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "RecoEgamma/EgammaTools/interface/HoECalculator.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/PhotonIdentification/interface/PhotonIsolationCalculator.h"

// PhotonCoreProducer inherits from EDProducer, so it can be a module:
class PhotonCoreProducer : public edm::EDProducer {

 public:

  PhotonCoreProducer (const edm::ParameterSet& ps);
  ~PhotonCoreProducer();

  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  void fillPhotonCollection(edm::Event& evt,
			    edm::EventSetup const & es,
                            const edm::Handle<reco::SuperClusterCollection> & scHandle,
			    const edm::Handle<reco::ConversionCollection> & conversionHandle,
			    const edm::Handle<reco::ElectronSeedCollection> & pixelSeeds,
			    reco::PhotonCoreCollection & outputCollection,
			    int& iSC);

  reco::ConversionRef solveAmbiguity( const edm::Handle<reco::ConversionCollection> & conversionHandle, reco::SuperClusterRef& sc);



  std::string PhotonCoreCollection_;
  edm::InputTag scHybridBarrelProducer_;
  edm::InputTag scIslandEndcapProducer_;
  edm::InputTag scHybridBarrelCollection_;
  edm::InputTag scIslandEndcapCollection_;

  edm::InputTag conversionProducer_;

  double minSCEt_;
  bool validConversions_;
  std::string pixelSeedProducer_;
  edm::ParameterSet conf_;
  bool validPixelSeeds_;
  bool risolveAmbiguity_;

};
#endif
