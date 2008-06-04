// -*- C++ -*-
//
// Package:   EcalHighEnCosmicFilter
// Class:     EcalHighEnCosmicFilter
//
//class EcalHighEnCosmicFilter EcalHighEnCosmicFilter.cc
//
// Original Author:  Serena OGGERO
//         Created:  We May 14 10:10:52 CEST 2008


#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "CaloOnlineTools/EcalTools/plugins/EcalHighEnCosmicFilter.h"

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

using namespace edm;
using namespace std;
using namespace reco;

//
EcalHighEnCosmicFilter::EcalHighEnCosmicFilter(const edm::ParameterSet& iConfig)
{
  BarrelClusterCollection = iConfig.getParameter<edm::InputTag>("barrelClusterCollection");
  EndcapClusterCollection = iConfig.getParameter<edm::InputTag>("endcapClusterCollection");

  EnergyCut = iConfig.getUntrackedParameter<double>("energycut");
  nHighEnClus = 0;
  nEvent = 0;
  nGoodEvent = 0;
  //edm::LogVerbatim("") <<  "BEGIN JOB ------------------ ";
  //cout << "BEGIN JOB" << endl;
}

EcalHighEnCosmicFilter::~EcalHighEnCosmicFilter()
{
  cout << "-------------- " << endl << endl << "Nevent processed : " << nEvent ;
  cout << endl << "Nevents with high energy deposits : " << nGoodEvent << endl;
  cout << "-------------- " << endl;

  // edm::LogVerbatim("") << "End Job ------------------- " ;
  // edm::LogVerbatim("") << "Nevents processed : " << nEvent ;
  // edm::LogVerbatim("") << "Nevents with high energy deposits: " <<nGoodEvent ;
}

bool EcalHighEnCosmicFilter::filter( edm::Event& iEvent, const edm::EventSetup& iSetup)
{
 int ievt = iEvent.id().event();
 nEvent++;
 // edm::LogVerbatim("") << "handling cluster collection " ;
 Handle<reco::BasicClusterCollection> bccHandle;
 Handle<reco::BasicClusterCollection> eccHandle;

 iEvent.getByLabel(BarrelClusterCollection, bccHandle);
 if (!(bccHandle.isValid()))
 {
   LogWarning("EcalHighEnCosmicFilter") << BarrelClusterCollection << " not available in event " << ievt;
   //return;
 }
 else 
 {
   edm::LogVerbatim("") << "I took the right barrel collection" ;
 }
   //LogDebug("EcalHighEnCosmicFilter") << "event " << ievt;

 iEvent.getByLabel(EndcapClusterCollection, eccHandle);
 if (!(eccHandle.isValid()))
 {
   //LogWarning("EcalHighEnCosmicFilter") << EndcapClusterCollection << " not available";
   //return;
 }
 else 
 {
   //edm::LogVerbatim("") << "I took the right endcap collection " ;
 }


 bool accepted = false;

 const reco::BasicClusterCollection *clusterCollection_p = bccHandle.product();
 for (reco::BasicClusterCollection::const_iterator clus = clusterCollection_p->begin(); clus != clusterCollection_p->end(); ++clus)
 {
  double energy = clus->energy();
  if ( energy >= EnergyCut )
  {
   //edm::LogVerbatim("") <<  "Cluster with energy " << energy ;
   nHighEnClus++;
   nGoodEvent++;
   accepted = true;
   break;
  }
 }
 
 return accepted;
}

