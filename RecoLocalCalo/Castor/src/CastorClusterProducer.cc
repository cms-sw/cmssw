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
// $Id: CastorClusterProducer.cc,v 1.4 2010/01/25 13:35:12 vlimant Exp $
//
//

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
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/Jet.h"

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
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      double phiangle (double testphi);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      typedef math::XYZPointD Point;
      typedef ROOT::Math::RhoEtaPhiPoint TowerPoint;
      typedef ROOT::Math::RhoZPhiPoint CellPoint;
      typedef std::vector<reco::CastorCell> CastorCellCollection;
      typedef std::vector<reco::CastorTower> CastorTowerCollection;
      typedef std::vector<reco::CastorCluster> CastorClusterCollection;
      std::string input_, basicjets_;
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
  input_(iConfig.getUntrackedParameter<std::string>("inputtowers","")),
  basicjets_(iConfig.getUntrackedParameter<std::string>("basicjets","")),
  ktalgo_(iConfig.getUntrackedParameter<bool>("KtAlgo",false)),
  clusteralgo_(iConfig.getUntrackedParameter<bool>("ClusterAlgo",false)),
  recom_(iConfig.getUntrackedParameter<unsigned int>("KtRecombination",2)),
  rParameter_(iConfig.getUntrackedParameter<double>("KtrParameter",1.))
{
  // register your products
  /*if (ktalgo_ == true)*/ produces<CastorClusterCollection>();
  //if (clusteralgo_ == true) produces<CastorClusterCollection>();
  //if (basicjets_ == "CastorFastjetRecoKt") produces<CastorClusterCollection>();
  //if (basicjets_ == "CastorFastjetRecoSISCone") produces<CastorClusterCollection>();
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
  
  LogDebug("CastorClusterProducer")
    <<"3. entering CastorClusterProducer";
  
  if ( input_ != "") {
  
  // Produce CastorClusters from CastorTowers
  
  edm::Handle<CastorTowerCollection> InputTowers;
  iEvent.getByLabel(input_, InputTowers);

  auto_ptr<CastorClusterCollection> OutputClustersfromKtAlgo (new CastorClusterCollection);
  auto_ptr<CastorClusterCollection> OutputClustersfromClusterAlgo (new CastorClusterCollection);
  
  // get and check input size
  int nTowers = InputTowers->size();

  if (nTowers==0) LogDebug("CastorClusterProducer")<<"Warning: You are trying to run the Cluster algorithm with 0 input towers.";

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

    iEvent.put(OutputClustersfromKtAlgo);
  } // end build cluster from KtAlgo
    
  // build cluster from ClusterAlgo
  if (clusteralgo_ == true) {
    // code
    iEvent.put(OutputClustersfromClusterAlgo);
  }
  
  }
  
  if ( basicjets_ != "") {
  
  	//cout << " entering the basicjet --> cluster code " << endl;
  
  	Handle<BasicJetCollection> bjCollection;
   	iEvent.getByLabel(basicjets_,bjCollection);
	
	Handle<CastorTowerCollection> ctCollection;
	iEvent.getByLabel("CastorTowerReco",ctCollection);
	
	auto_ptr<CastorClusterCollection> OutputClustersfromBasicJets (new CastorClusterCollection);
	
	if (bjCollection->size()==0) LogDebug("CastorClusterProducer")<< "Warning: You are trying to run the Cluster algorithm with 0 input basicjets.";
   
   	for (unsigned i=0; i< bjCollection->size();i++) {
   		const BasicJet* bj = &(*bjCollection)[i];
		
		double energy = bj->energy();
		TowerPoint temp(88.5,bj->eta(),bj->phi());
  		Point position(temp);
		double emEnergy = 0.;
		double hadEnergy = 0.;
		double width = 0.;
		double depth = 0.;
		double fhot = 0.;
		double sigmaz = 0.;
		CastorTowerRefVector usedTowers;
		double zmean = 0.;
		double z2mean = 0.;
	
		vector<CandidatePtr> ccp = bj->getJetConstituents();
		vector<CandidatePtr>::const_iterator itParticle;
   		for (itParticle=ccp.begin();itParticle!=ccp.end();++itParticle){	    
        		const CastorTower* castorcand = dynamic_cast<const CastorTower*>(itParticle->get());
			//cout << " castortowercandidate reference energy = " << castorcand->castorTower()->energy() << endl;
			//cout << " castortowercandidate reference eta = " << castorcand->castorTower()->eta() << endl;
			//cout << " castortowercandidate reference phi = " << castorcand->castorTower()->phi() << endl;
			//cout << " castortowercandidate reference depth = " << castorcand->castorTower()->depth() << endl;
			
			//CastorTowerCollection *ctc = new CastorTowerCollection();
			//ctc->push_back(*castorcand);
			//CastorTowerRef towerref = CastorTowerRef(ctc,0);
			
			size_t thisone = 0;
			for (size_t l=0;l<ctCollection->size();l++) {
				const CastorTower ct = (*ctCollection)[l];
				if ( abs(ct.phi() - castorcand->phi()) < 0.0001 ) { thisone = l;}
			}
			
			CastorTowerRef towerref(ctCollection,thisone); 
			usedTowers.push_back(towerref);
			emEnergy += castorcand->emEnergy();
			hadEnergy += castorcand->hadEnergy();
			depth += castorcand->depth()*castorcand->energy();
			width += pow(phiangle(castorcand->phi() - bj->phi()),2)*castorcand->energy();
      			fhot += castorcand->fhot()*castorcand->energy();
			
			// loop over cells
      			for (CastorCell_iterator it = castorcand->cellsBegin(); it != castorcand->cellsEnd(); it++) {
				CastorCellRef cell_p = *it;
				Point rcell = cell_p->position();
				double Ecell = cell_p->energy();
				zmean += Ecell*cell_p->z();
				z2mean += Ecell*cell_p->z()*cell_p->z();
      			} // end loop over cells
		}
		//cout << "" << endl;
		
		depth = depth/energy;
		width = sqrt(width/energy);
		fhot = fhot/energy;
		
		zmean = zmean/energy;
    		z2mean = z2mean/energy;
    		double sigmaz2 = z2mean - zmean*zmean;
    		if(sigmaz2 > 0) sigmaz = sqrt(sigmaz2);
		
		CastorCluster cc(energy,position,emEnergy,hadEnergy,emEnergy/energy,width,depth,fhot,sigmaz,usedTowers);
		OutputClustersfromBasicJets->push_back(cc);
   	}
	
	iEvent.put(OutputClustersfromBasicJets);
  
  }
 
}

// help function to calculate phi within [-pi,+pi]
double CastorClusterProducer::phiangle (double testphi) {
  double phi = testphi;
  while (phi>M_PI) phi -= (2*M_PI);
  while (phi<-M_PI) phi += (2*M_PI);
  return phi;
}

// ------------ method called once each job just before starting event loop  ------------
void CastorClusterProducer::beginJob() {
  LogDebug("CastorClusterProducer")
    <<"Starting CastorClusterProducer";
}

// ------------ method called once each job just after ending the event loop  ------------
void CastorClusterProducer::endJob() {
  LogDebug("CastorClusterProducer")
    <<"Ending CastorClusterProducer";
}

//define this as a plug-in
DEFINE_FWK_MODULE(CastorClusterProducer);
