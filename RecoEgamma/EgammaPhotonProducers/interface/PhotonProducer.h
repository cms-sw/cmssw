#ifndef RecoEgamma_EgammaPhotonProducers_PhotonProducer_h
#define RecoEgamma_EgammaPhotonProducers_PhotonProducer_h
/** \class PhotonProducer
 **  
 **
 **  $Id: PhotonProducer.h,v 1.5 2007/01/31 17:15:02 futyand Exp $ 
 **  $Date: 2007/01/31 17:15:02 $ 
 **  $Revision: 1.5 $
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
  std::string vertexProducer_;
  edm::ParameterSet conf_;


};
#endif
