// -*- C++ -*-
//
// Package:   EcalSkim
// Class:     EcalSkim
//
//class EcalSkim EcalSkim.cc
//
// Original Author:  Serena OGGERO
//         Created:  We May 14 10:10:52 CEST 2008
//        Modified:  Toyoko ORIMOTO

#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "DPGAnalysis/Skims/interface/EcalSkim.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

using namespace edm;
using namespace std;
using namespace reco;

EcalSkim::EcalSkim(const edm::ParameterSet& iConfig) {
  BarrelClusterCollection = iConfig.getParameter<edm::InputTag>("barrelClusterCollection");
  EndcapClusterCollection = iConfig.getParameter<edm::InputTag>("endcapClusterCollection");

  EnergyCutEB = iConfig.getUntrackedParameter<double>("energyCutEB");
  EnergyCutEE = iConfig.getUntrackedParameter<double>("energyCutEE");
}

EcalSkim::~EcalSkim() {}

bool EcalSkim::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  int ievt = iEvent.id().event();

  edm::Handle<reco::SuperClusterCollection> bccHandle;  // barrel
  edm::Handle<reco::SuperClusterCollection> eccHandle;  // endcap

  iEvent.getByLabel("cosmicSuperClusters", "CosmicBarrelSuperClusters", bccHandle);
  if (!(bccHandle.isValid())) {
    LogWarning("EcalSkim") << BarrelClusterCollection << " not available in event " << ievt;
    return false;
  } else {
    //   edm::LogVerbatim("") << "I took the right barrel collection" ;
  }
  iEvent.getByLabel("cosmicSuperClusters", "CosmicEndcapSuperClusters", eccHandle);

  if (!(eccHandle.isValid())) {
    LogWarning("EcalSkim") << EndcapClusterCollection << " not available";
    //return false;
  } else {
    //edm::LogVerbatim("") << "I took the right endcap collection " ;
  }

  bool accepted = false;
  bool acceptedEB = false;
  bool acceptedEE = false;

  // barrel
  const reco::SuperClusterCollection* clusterCollectionEB = bccHandle.product();
  for (reco::SuperClusterCollection::const_iterator clus = clusterCollectionEB->begin();
       clus != clusterCollectionEB->end();
       ++clus) {
    if (clus->energy() >= EnergyCutEB) {
      acceptedEB = true;
      break;
    }
  }

  // endcap
  const reco::SuperClusterCollection* clusterCollectionEE = eccHandle.product();
  for (reco::SuperClusterCollection::const_iterator clus = clusterCollectionEE->begin();
       clus != clusterCollectionEE->end();
       ++clus) {
    if (clus->energy() >= EnergyCutEE) {
      acceptedEE = true;
      break;
    }
  }

  // if there is at least one high energy cluster in EB OR EE, accept
  if (acceptedEB || acceptedEE)
    accepted = true;

  return accepted;
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalSkim);
