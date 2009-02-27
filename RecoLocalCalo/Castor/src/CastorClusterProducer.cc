// -*- C++ -*-
//
// Package:    Castor
// Class:      CastorClusterProducer
// 
/**\class CastorClusterProducer CastorClusterProducer.cc RecoLocalCalo/Castor/src/CastorClusterProducer.cc

 Description: CastorCluster Reconstruction Producer. Produces Clusters from Towers
 Implementation:
*/

//
// Original Author:  Hans Van Haevermaet, Benoit Roland
//         Created:  Wed Jul  9 14:00:40 CEST 2008
// $Id: Castor.cc,v 1.3 2008/12/09 08:44:01 hvanhaev Exp $
//
//

#define debug 0

// system include 
#include <memory>
#include <vector>
#include <iostream>
#include <TMath.h>
#include <TRandom3.h>

// user include 
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/Point3D.h"

// Castor Object include
#include "DataFormats/CastorReco/interface/CastorCell.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorCluster.h"

// Castor Reco include 
#include "RecoLocalCalo/Castor/interface/KtAlgorithm.h"

//
// class decleration
//

class CastorClusterProducer : public edm::EDProducer {
   public:
      explicit CastorClusterProducer(const edm::ParameterSet&);
      ~CastorClusterProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      typedef math::XYZPointD Point;
      typedef ROOT::Math::RhoEtaPhiPoint TowerPoint;
      typedef ROOT::Math::RhoZPhiPoint CellPoint;
      typedef std::vector<reco::CastorCell> CastorCellCollection;
      typedef std::vector<reco::CastorTower> CastorTowerCollection;
      typedef std::vector<reco::CastorCluster> CastorClusterCollection;
      std::string input_;
      bool ktalgo_;
      bool clusteralgo_;
      unsigned int recom_;
      double rParameter_;
};

//
// constants, enums and typedefs
//

const double MYR2D = 180/M_PI;

//
// static data member definitions
//

//
// constructor and destructor
//

CastorClusterProducer::CastorClusterProducer(const edm::ParameterSet& iConfig) :
  input_(iConfig.getUntrackedParameter<std::string>("inputprocess","CastorTowerReco")),
  ktalgo_(iConfig.getUntrackedParameter<bool>("KtAlgo",true)),
  clusteralgo_(iConfig.getUntrackedParameter<bool>("ClusterAlgo",false)),
  recom_(iConfig.getUntrackedParameter<unsigned int>("KtRecombination",2)),
  rParameter_(iConfig.getUntrackedParameter<double>("KtrParameter",1.))
{
  // register your products
  if (ktalgo_ == true) produces<CastorClusterCollection>("fromKtAlgo");
  if (clusteralgo_ == true) produces<CastorClusterCollection>("fromClusterAlgo");
  // now do what ever other initialization is needed
}


CastorClusterProducer::~CastorClusterProducer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void CastorClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  using namespace edm;
  using namespace reco;
  using namespace std;
  using namespace TMath;
  
  // Produce CastorClusters from CastorTowers
  
  edm::Handle<CastorTowerCollection> InputTowers;
  iEvent.getByLabel(input_, InputTowers);

  auto_ptr<CastorClusterCollection> OutputClustersfromKtAlgo (new CastorClusterCollection);
  auto_ptr<CastorClusterCollection> OutputClustersfromClusterAlgo (new CastorClusterCollection);
  
  // get and check input size
  int nTowers = InputTowers->size();
  
  if(debug) cout<<""<<endl;
  if(debug) cout<<"---------------------------------"<<endl;
  if(debug) cout<<"3. entering CastorClusterProducer"<<endl;
  if(debug) cout<<"---------------------------------"<<endl;
  if(debug) cout<<""<<endl;

  if (nTowers==0) cout<<"Warning: You are trying to run the Cluster algorithm with 0 input towers. \n";

  CastorTowerRefVector posInputTowers, negInputTowers;

  for (size_t i = 0; i < InputTowers->size(); ++i) {
    reco::CastorTowerRef tower_p = reco::CastorTowerRef(InputTowers, i);
    if (tower_p->eta() > 0.) posInputTowers.push_back(tower_p);
    if (tower_p->eta() < 0.) negInputTowers.push_back(tower_p);
  }
  
  // build cluster from KtAlgo
  if (ktalgo_ == true) {
    KtAlgorithm ktalgo;
    CastorClusterCollection posClusters, negClusters;

    if (posInputTowers.size() != 0) {
      posClusters = ktalgo.runKtAlgo(posInputTowers,recom_,rParameter_);
      for (size_t i=0; i<posClusters.size(); i++) OutputClustersfromKtAlgo->push_back(posClusters[i]);
    }

    if (negInputTowers.size() != 0) {
      negClusters = ktalgo.runKtAlgo(negInputTowers,recom_,rParameter_);
      for (size_t i=0; i<negClusters.size(); i++) OutputClustersfromKtAlgo->push_back(negClusters[i]);
    }

    iEvent.put(OutputClustersfromKtAlgo,"fromKtAlgo");
  } // end build cluster from KtAlgo
    
  // build cluster from ClusterAlgo
  if (clusteralgo_ == true) {
    // code
    iEvent.put(OutputClustersfromClusterAlgo,"fromClusterAlgo");
  }
 
}

// ------------ method called once each job just before starting event loop  ------------
void CastorClusterProducer::beginJob(const edm::EventSetup&) {
if(debug) std::cout<<"Starting CastorClusterProducer"<<std::endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void CastorClusterProducer::endJob() {
if(debug) std::cout<<"Ending CastorClusterProducer"<<std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CastorClusterProducer);
