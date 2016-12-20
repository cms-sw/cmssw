#ifndef RECOEGAMMA_EGAMMAISOLATIONALGOS_EGAMMAISOESDETIDCOLLECTIONPRODUCER_H
#define RECOEGAMMA_EGAMMAISOLATIONALGOS_EGAMMAISOESDETIDCOLLECTIONPRODUCER_H


// -*- C++ -*-
//
// Package:    EgammaIsoESDetIdCollectionProducer
// Class:      EgammaIsoESDetIdCollectionProducer
// 
/**\class EgammaIsoESDetIdCollectionProducer 

author: Sam Harper (inspired by InterestingDetIdProducer)
 
Make a collection of detids to be kept in a AOD rechit collection
These are all the ES DetIds of ES PFClusters associated to all PF clusters within dR of ele/pho/sc
The aim is to save enough preshower info in the AOD to remake the PF clusters near an ele/pho/sc
*/



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

class EgammaIsoESDetIdCollectionProducer : public edm::stream::EDProducer<> {
public:
  //! ctor
  explicit EgammaIsoESDetIdCollectionProducer(const edm::ParameterSet&);
  virtual void beginRun (edm::Run const&, const edm::EventSetup&) override final;
  //! producer
  virtual void produce(edm::Event &, const edm::EventSetup&);

private:
  void addDetIds(const reco::SuperCluster& superClus,reco::PFClusterCollection clusters,const reco::PFCluster::EEtoPSAssociation& eeClusToESMap,std::vector<DetId>& detIdsToStore);

  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::PFCluster::EEtoPSAssociation>         eeClusToESMapToken_;
  edm::EDGetTokenT<reco::PFClusterCollection> ecalPFClustersToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> superClustersToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> elesToken_;
  edm::EDGetTokenT<reco::PhotonCollection> phosToken_;

  std::string interestingDetIdCollection_;
 
  float minSCEt_;
  float minEleEt_;
  float minPhoEt_;

  float maxDR_;
    
};

#endif
