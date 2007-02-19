#ifndef RecoEgamma_EgammaPhotonProducers_PhotonProducer_h
#define RecoEgamma_EgammaPhotonProducers_PhotonProducer_h
/** \class PhotonProducer
 **  
 **
 **  $Id: PhotonProducer.h,v 1.6 2007/02/14 23:41:38 futyand Exp $ 
 **  $Date: 2007/02/14 23:41:38 $ 
 **  $Revision: 1.6 $
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"

// PhotonProducer inherits from EDProducer, so it can be a module:
class PhotonProducer : public edm::EDProducer {

 public:

  PhotonProducer (const edm::ParameterSet& ps);
  ~PhotonProducer();

  virtual void beginJob (edm::EventSetup const & es);
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  void fillPhotonCollection(const edm::Handle<reco::SuperClusterCollection> & scHandle,
			    const reco::BasicClusterShapeAssociationCollection& clshpMap,
			    const CaloSubdetectorGeometry *geometry,
			    const CaloSubdetectorGeometry *geometryES,
			    const EcalRecHitCollection *hits,
			    const reco::SeedSuperClusterAssociationCollection& pixelSeedAssoc,
			    math::XYZPoint & vtx,
			    reco::PhotonCollection & outputCollection,
			    int iSC);

  std::string PhotonCollection_;
  std::string scHybridBarrelProducer_;
  std::string scIslandEndcapProducer_;
  std::string scHybridBarrelCollection_;
  std::string scIslandEndcapCollection_;
  std::string barrelClusterShapeMapProducer_;
  std::string barrelClusterShapeMapCollection_;
  std::string endcapClusterShapeMapProducer_;
  std::string endcapClusterShapeMapCollection_;
  std::string barrelHitProducer_;
  std::string endcapHitProducer_;
  std::string barrelHitCollection_;
  std::string endcapHitCollection_;
  std::string pixelSeedAssocProducer_;
  std::string vertexProducer_;
  edm::ParameterSet conf_;

  PositionCalc posCalculator_;

};
#endif
