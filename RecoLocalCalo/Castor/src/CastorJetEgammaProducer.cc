// -*- C++ -*-
//
// Package:    Castor
// Class:      CastorJetEgammaProducer
// 
/**\class CastorJetEgammaProducer CastorJetEgammaProducer.cc RecoLocalCalo/Castor/src/CastorJetEgammaProducer.cc

 Description: CastorJet/Egamma Reconstruction Producer. Produces Jets and Egammas from Clusters

 Implementation:
     <Notes on implementation>
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
#include "DataFormats/CastorReco/interface/CastorEgamma.h"
#include "DataFormats/CastorReco/interface/CastorCluster.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"

//
// class declaration
//

class CastorJetEgammaProducer : public edm::EDProducer {
   public:
      explicit CastorJetEgammaProducer(const edm::ParameterSet&);
      ~CastorJetEgammaProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void select(const reco::CastorClusterRefVector& InputClusters, reco::CastorJetCollection& jets, reco::CastorEgammaCollection& egammas);

      // ----------member data ---------------------------
      typedef math::XYZPointD Point;
      typedef ROOT::Math::RhoEtaPhiPoint TowerPoint;
      typedef ROOT::Math::RhoZPhiPoint CellPoint;
      typedef std::vector<reco::CastorCluster> CastorClusterCollection;
      typedef std::vector<reco::CastorJet> CastorJetCollection;
      typedef std::vector<reco::CastorEgamma> CastorEgammaCollection;
      typedef edm::RefVector<reco::CastorClusterCollection> CastorClusterRefVector;
      std::string input_;
      bool fastsim_;
      bool ktalgo_;
      bool clusteralgo_;
};

//
// constants, enums and typedefs
//

const double MYR2D = 180/M_PI;

//
// static data member definitions
//

//
// constructors and destructor
//

CastorJetEgammaProducer::CastorJetEgammaProducer(const edm::ParameterSet& iConfig) :
	input_(iConfig.getUntrackedParameter<std::string>("inputprocess","CastorClusterReco")),
	fastsim_(iConfig.getUntrackedParameter<bool>("fastsim",false)),
	ktalgo_(iConfig.getUntrackedParameter<bool>("KtAlgo",true)),
	clusteralgo_(iConfig.getUntrackedParameter<bool>("ClusterAlgo",false))

{
  //register your products
  if (ktalgo_) {
    produces<CastorEgammaCollection>("fromKtAlgo");
    produces<CastorJetCollection>("fromKtAlgo");
  }
  if (clusteralgo_) {
    produces<CastorEgammaCollection>("fromClusterAlgo");
    produces<CastorJetCollection>("fromClusterAlgo");
  }
   
   //now do what ever other initialization is needed
}


CastorJetEgammaProducer::~CastorJetEgammaProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void CastorJetEgammaProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  using namespace edm;
  using namespace reco;
  using namespace std;
  using namespace TMath;
  
  // Produce CastorJets and CastorEgammas from CastorClusters
    
  if (ktalgo_) {
    
    edm::Handle<CastorClusterCollection> InputClusters;
    iEvent.getByLabel(input_,"fromKtAlgo",InputClusters);

    auto_ptr<CastorJetCollection> OutputJets (new CastorJetCollection);
    auto_ptr<CastorEgammaCollection> OutputEgammas (new CastorEgammaCollection);

    // get and check input size
    int nClusters = InputClusters->size();

    if(debug) cout<<""<<endl;
    if(debug) cout<<"-----------------------------------"<<endl;
    if(debug) cout<<"4. entering CastorJetEgammaProducer"<<endl;
    if(debug) cout<<"-----------------------------------"<<endl;
    if(debug) cout<<""<<endl;

    if (nClusters==0) cout<<"Warning: You are trying to run the Jet Egamma algorithm with 0 input clusters. \n";
    
    CastorClusterRefVector Clusters;
    
    for (size_t i = 0; i < InputClusters->size(); ++i) {
      reco::CastorClusterRef cluster_p = reco::CastorClusterRef(InputClusters, i);
      Clusters.push_back(cluster_p);
    }

    CastorJetCollection Jets;
    CastorEgammaCollection Egammas;

    select(Clusters,Jets,Egammas);

    for (size_t i=0; i<Jets.size(); i++) OutputJets->push_back(Jets[i]);
    for (size_t i=0; i<Egammas.size(); i++) OutputEgammas->push_back(Egammas[i]); 

    iEvent.put(OutputJets,"fromKtAlgo");
    iEvent.put(OutputEgammas,"fromKtAlgo");
  }
  
  if (clusteralgo_) {
    
    edm::Handle<CastorClusterCollection> InputClusters;
    iEvent.getByLabel(input_,"fromClusterAlgo",InputClusters);

    auto_ptr<CastorJetCollection> OutputJets (new CastorJetCollection);
    auto_ptr<CastorEgammaCollection> OutputEgammas (new CastorEgammaCollection);

    // get and check input size
    int nClusters = InputClusters->size();

    if(debug) cout<<""<<endl;
    if(debug) cout<<"-----------------------------------"<<endl;
    if(debug) cout<<"4. entering CastorJetEgammaProducer"<<endl;
    if(debug) cout<<"-----------------------------------"<<endl;
    if(debug) cout<<""<<endl;

    if (nClusters==0) cout<<"Warning: You are trying to run the Jet Egamma algorithm with 0 input clusters. \n";

    CastorClusterRefVector Clusters;

    for (size_t i = 0; i < InputClusters->size(); ++i) {
      reco::CastorClusterRef cluster_p = reco::CastorClusterRef(InputClusters, i);
      Clusters.push_back(cluster_p);
    }

    CastorJetCollection Jets;
    CastorEgammaCollection Egammas;

    select(Clusters,Jets,Egammas);

    for (size_t i=0; i<Jets.size(); i++) OutputJets->push_back(Jets[i]);
    for (size_t i=0; i<Egammas.size(); i++) OutputEgammas->push_back(Egammas[i]); 

    iEvent.put(OutputJets,"fromClusterAlgo");
    iEvent.put(OutputEgammas,"fromClusterAlgo");
  }
}

// ------------ method called once each job just before starting event loop  ------------
void CastorJetEgammaProducer::beginJob(const edm::EventSetup&) {
  if(debug) std::cout<<"Starting CastorJetEgammaProducer"<<std::endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void CastorJetEgammaProducer::endJob() {
  if(debug) std::cout<<"Ending CastorJetEgammaProducer"<<std::endl;
}

void CastorJetEgammaProducer::select(const reco::CastorClusterRefVector& InputClusters, reco::CastorJetCollection& jets,
				     reco::CastorEgammaCollection& egammas) {

  using namespace reco;
  using namespace std;

  if(debug) cout<<""<<endl;
  if(debug) cout<<"-------------------------------"<<endl;
  if(debug) cout<<"entering Select Jet Egamma code"<<endl;
  if(debug) cout<<"-------------------------------"<<endl;
  if(debug) cout<<""<<endl;

  // loop over clusters
  for (CastorCluster_iterator it_cluster = (InputClusters).begin(); it_cluster != (InputClusters).end(); it_cluster++) {

    reco::CastorClusterRef cluster_p = *it_cluster;

    bool IDpion = true;
    bool IDelectron = true;
	
    // pion cuts
    if (TMath::Abs(cluster_p->depth()) < 14450 && cluster_p->energy() < 175.) IDpion = false;
    if (TMath::Abs(cluster_p->depth()) < 14460 && cluster_p->energy() > 175.) IDpion = false;
    if (cluster_p->fem() > 0.95) IDpion = false;

    if (IDpion) {
        // cluster identified as jet/pion, correct the energy in full sim chain
	double energycal;
	if ( !fastsim_ ) {
		double correction_factor = 0.6148 + 0.0504*log(cluster_p->energy());
		energycal = cluster_p->energy()/correction_factor;
	} else {
		energycal = cluster_p->energy();
	}
        jets.push_back(CastorJet(energycal,cluster_p));
    }
	
    // electron cuts	
    if (cluster_p->sigmaz() > 30. && cluster_p->energy() < 75.) IDelectron = false;
    if (cluster_p->sigmaz() > 40. && cluster_p->energy() > 75.) IDelectron = false;
    if (TMath::Abs(cluster_p->depth()) > 14450 && cluster_p->energy() < 125.) IDelectron = false;
    if (TMath::Abs(cluster_p->depth()) > 14460 && cluster_p->energy() > 125.) IDelectron = false;
    if (cluster_p->fhot() < 0.45) IDelectron = false;
    if (cluster_p->fem() < 0.9) IDelectron = false;
    if (cluster_p->width() > 0.2) IDelectron = false; 
	 
    if (IDelectron && !IDpion) {
      // cluster identified as egamma object, just copy it
      egammas.push_back(CastorEgamma(cluster_p->energy(),cluster_p));
    }
 
  } //end loop over clusters
    
  return;

}

//define this as a plug-in
DEFINE_FWK_MODULE(CastorJetEgammaProducer);
