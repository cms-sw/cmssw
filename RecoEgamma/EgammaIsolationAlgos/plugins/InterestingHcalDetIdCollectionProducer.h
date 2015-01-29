#ifndef _INTERESTINGHCALDETIDCOLLECTIONPRODUCER_H
#define _INTERESTINGHCALDETIDCOLLECTIONPRODUCER_H

// -*- C++ -*-
//
// Package:    InterestingHcalDetIdCollectionProducer
// Class:      InterestingHcalDetIdCollectionProducer
// 
/**\class InterestingHcalDetIdCollectionProducer 
Original author: Paolo Meridiani PH/CMG
 
Make a collection of detids to be kept tipically in a AOD rechit collection
Modified from the ECAL version "InterestingHcalDetIdCollectionProducer" to be HCAL

The following classes of "interesting id" are considered

    1.in a region around  the seed of the cluster collection specified
      by paramter basicClusters. The size of the region is specified by
      minimalEtaSize_, minimalPhiSize_
 
    2. if the severity of the hit is >= severityLevel_
       If severityLevel=0 this class is ignored

    3. Channels next to dead ones,  keepNextToDead_ is true
    4. Channels next to the EB/EE transition if keepNextToBoundary_ is true
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

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"



class InterestingHcalDetIdCollectionProducer : public edm::stream::EDProducer<> {
public:
  //! ctor
  explicit InterestingHcalDetIdCollectionProducer(const edm::ParameterSet&);
  virtual void beginRun (edm::Run const&, const edm::EventSetup&) override final;
  //! producer
  virtual void produce(edm::Event &, const edm::EventSetup&);

private:
  void addDetIds(const reco::SuperCluster& superClus,const HBHERecHitCollection& recHits,std::vector<DetId>& detIdsToStore);

  // ----------member data ---------------------------
  edm::EDGetTokenT<HBHERecHitCollection>         recHitsToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> superClustersToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> elesToken_;
  edm::EDGetTokenT<reco::PhotonCollection> phosToken_;

  std::string interestingDetIdCollection_;
 
  float minSCEt_;
  float minEleEt_;
  float minPhoEt_;

  int maxDIEta_;
  int maxDIPhi_;
  
  edm::ESHandle<CaloTowerConstituentsMap> towerMap_;
  
};

#endif
