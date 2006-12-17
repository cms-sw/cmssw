// -*- C++ -*-
//
// Package:     EgammaPhotonProducers
// Class  :     PhotonCorrectionProd
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Thu May 25 11:10:50 CDT 2006
// $Id$
//

#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonCorrectionProducer.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
  
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/EtaCorrectionAlgo.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/E1E9CorrectionAlgo.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/E9ESCCorrectionAlgo.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonBasketBorderCorrectionAlgo.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

PhotonCorrectionProducer::PhotonCorrectionProducer(const edm::ParameterSet& ps)
{
  registorAlgos();
  photonProducer_         = ps.getParameter<std::string>("photonProducer");
  photonCollection_       = ps.getParameter<std::string>("photonCollection");
  photonCorrCollection_   = ps.getParameter<std::string>("photonCorrCollection");
  algoCollection_         = ps.getParameter<std::string>("algoCollection");
  barrelClusterShapeMapProducer_   = ps.getParameter<std::string>("barrelClusterShapeMapProducer");
  barrelClusterShapeMapCollection_ = ps.getParameter<std::string>("barrelClusterShapeMapCollection");
  endcapClusterShapeMapProducer_   = ps.getParameter<std::string>("endcapClusterShapeMapProducer");
  endcapClusterShapeMapCollection_ = ps.getParameter<std::string>("endcapClusterShapeMapCollection");


  std::string word;
  std::stringstream ss(algoCollection_);
  while(ss>>word){
    if(algo_m.find(word) != algo_m.end()){
      algo_v.push_back(algo_m[word]);      
    }
    else {
      std::string errorMsg = std::string("PhotonCorrectionProducer: Can not find energy correction alogrithm ") + word + std::string(".");
      throw std::runtime_error(errorMsg);
    }
  }    

  produces<reco::PhotonCollection>(photonCorrCollection_);
}

PhotonCorrectionProducer::~PhotonCorrectionProducer()
{
  clearAlgos();
}

void PhotonCorrectionProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<reco::PhotonCollection> photonHandle;
  try
    {
      evt.getByLabel(photonProducer_, photonCollection_, photonHandle);    
    } 
  catch (cms::Exception&ex)
    {
      edm::LogError("PhotonCorrectionProducer") << "Error! Can't get the product "<<photonCollection_.c_str();
    }

  
  const reco::PhotonCollection& photonCollection = *(photonHandle.product());
  std::auto_ptr<reco::PhotonCollection> photon_ap(new reco::PhotonCollection);

  edm::Handle<reco::BasicClusterShapeAssociationCollection> barrelClShpHandle;
  try{
    evt.getByLabel(barrelClusterShapeMapProducer_, barrelClusterShapeMapCollection_, barrelClShpHandle);
    }
  catch (cms::Exception&ex)
    {
      edm::LogError("PhotonCorrectionProducer") << "Error! Can't get the product "<<barrelClusterShapeMapCollection_.c_str();
    }

  edm::Handle<reco::BasicClusterShapeAssociationCollection> endcapClShpHandle;
  try{
    evt.getByLabel(endcapClusterShapeMapProducer_, endcapClusterShapeMapCollection_, endcapClShpHandle);
    }
  catch (cms::Exception&ex)
    {
      edm::LogError("PhotonCorrectionProducer") << "Error! Can't get the product "<<endcapClusterShapeMapCollection_.c_str();
    }
       
  for (reco::PhotonCollection::const_iterator phItr = photonCollection.begin(); phItr != photonCollection.end(); phItr++)
    {
      reco::Photon corPhoton = *phItr;
      for(std::vector<PhotonCorrectionAlgoBase*>::const_iterator algoItr = algo_v.begin(); algoItr != algo_v.end(); algoItr++)
	{
	  DetId id = corPhoton.superCluster()->seed()->getHitsByDetId()[0];
	  bool isBarrel = (id.subdetId() == EcalBarrel);
	  const reco::BasicClusterShapeAssociationCollection& clusterShapeCollection = isBarrel ? (*barrelClShpHandle) : (*endcapClShpHandle);
	  reco::Photon newPhotonCand =  isBarrel ? (*algoItr)->applyBarrelCorrection(corPhoton, clusterShapeCollection) : (*algoItr)->applyEndcapCorrection(corPhoton, clusterShapeCollection);;
	  corPhoton = newPhotonCand;
	}      
      photon_ap->push_back(corPhoton);       
    }
  
  evt.put(photon_ap, photonCorrCollection_);
  
}

void PhotonCorrectionProducer::registorAlgos()
{
  //register all algorithms
  algo_m["E9ESC"] = new E9ESCCorrectionAlgo;
  algo_m["Eta"]   = new EtaCorrectionAlgo;
  algo_m["E1E9"]  = new E1E9CorrectionAlgo;
  algo_m["BasketBorder"]  = new PhotonBasketBorderCorrectionAlgo;
}

void PhotonCorrectionProducer::clearAlgos()
{
  for(std::map<std::string, PhotonCorrectionAlgoBase*>::const_iterator itr = algo_m.begin(); itr != algo_m.end();itr++){
    delete (*itr).second;
  }
  algo_m.clear();
}
