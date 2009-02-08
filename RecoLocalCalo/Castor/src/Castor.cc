// -*- C++ -*-
//
// Package:    Castor
// Class:      Castor
// 
/**\class Castor Castor.cc RecoLocalCalo/Castor/src/Castor.cc

 Description: Castor Reconstruction Producer. Produces Cells, Towers, Jet and Egammas from CastorRecHits
              or Jets and Egammas from CastorTowers made in FastSimulation.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Hans Van Haevermaet
//         Created:  Wed Jul  9 14:00:40 CEST 2008
// $Id: Castor.cc,v 1.3 2008/12/09 08:44:01 hvanhaev Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <iostream>
#include <TMath.h>
#include <TRandom3.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/Point3D.h"

// Castorobject includes
#include "DataFormats/CastorReco/interface/CastorCell.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorEgamma.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

// Castor Kt algorithm include 
#include "RecoLocalCalo/Castor/interface/KtAlgorithm.h"
#include "RecoLocalCalo/Castor/interface/Egamma.h"
#include "RecoLocalCalo/Castor/interface/Tower.h"



//
// class decleration
//

class Castor : public edm::EDProducer {
   public:
      explicit Castor(const edm::ParameterSet&);
      ~Castor();

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
      typedef std::vector<reco::CastorEgamma> CastorEgammaCollection;
      typedef std::vector<reco::CastorJet> CastorJetCollection;
      bool FullReco_;
      unsigned int recom_;
      double rParameter_;
      double minRatio_;
      double maxWidth_;
      double maxDepth_;
      double towercut_;
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
Castor::Castor(const edm::ParameterSet& iConfig) :
	FullReco_(iConfig.getUntrackedParameter<bool>("FullReco",false)),
	recom_(iConfig.getUntrackedParameter<unsigned int>("KtRecombination",2)),
	rParameter_(iConfig.getUntrackedParameter<double>("KtrParameter",1.)),
	minRatio_(iConfig.getUntrackedParameter<double>("Egamma_minRatio",0.5)),
	maxWidth_(iConfig.getUntrackedParameter<double>("Egamma_maxWidth",0.2)),
	maxDepth_(iConfig.getUntrackedParameter<double>("Egamma_maxDepth",14488)),
	towercut_(iConfig.getUntrackedParameter<double>("towercut",0.))
{
   //register your products
   if (FullReco_ == true) {
   	produces<CastorCellCollection>();
   	produces<CastorTowerCollection>();
   	produces<CastorEgammaCollection>();
   	produces<CastorJetCollection>("fromKtAlgo");
   } else {
   	produces<CastorEgammaCollection>();
        produces<CastorJetCollection>("fromKtAlgo");	
   }
   
   //now do what ever other initialization is needed
  
}


Castor::~Castor()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
Castor::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   using namespace std;
   using namespace TMath;
   
   CastorTowerCollection posTowers, negTowers;
   
   // make difference between FullReco and FastReco(only making Jets & Egammas)
   if (FullReco_ == true) {
   
   // Produce CastorCells from CastorRecHits
   
   Handle<CastorRecHitCollection> castorRecHits;
   iEvent.getByLabel("castorreco", castorRecHits);
   
   auto_ptr<CastorCellCollection> CastorCells (new CastorCellCollection);
   CastorCellCollection posCells, negCells;
   
   // looping over all RecHits
   for (size_t i = 0; i < castorRecHits->size(); ++i) {
   	// get properties
   	const CastorRecHit & rh = (*castorRecHits)[i];
	//int section = rh.id().section();
	int sector = rh.id().sector();
	int module = rh.id().module();
	//double energy = rh.energy();
	int zside = rh.id().zside();
	//cout << "CastorRecHit in section " << section << " and sector " << sector << " and module " << module << " with energy " << energy << "\n"; 
   	
	// define CastorCell properties
	double zCell;
	double phiCell;
	
	// set z position of the cell
	if (module < 3) {
		// starting in EM section
		if ( module == 1) zCell = 14415;
		if (module == 2) zCell = 14464; 
	} else {
		// starting in HAD section
		zCell = 14534 + (module - 3)*92;
	}
	
	// set phi position of the cell
	double castorphis [16];
	for (int j = 0; j < 16; j++) {
		castorphis[j] = -2.94524 + j*0.3927;
        }
	if (sector > 8) {
		phiCell = castorphis[sector - 9];
	} else {
		phiCell = castorphis[sector + 7];
	}
	
	// add conditions to select in eta sides
	if (zside <= 0) zCell = -1*zCell;
	
	// store cell
	CellPoint tempcellposition(1.0,zCell,phiCell);
	Point cellposition(tempcellposition);
	if (rh.energy() > 0.) {
		CastorCell newCell(rh.energy(),cellposition);
		CastorCells->push_back(newCell);
		if (zCell < 0.) {
			negCells.push_back(newCell);
		} else {
			posCells.push_back(newCell);
		}
	}
	
   }
   
   iEvent.put(CastorCells);
   
   // Produce CastorTowers from CastorCells
   auto_ptr<CastorTowerCollection> CastorTowers (new CastorTowerCollection);
   //CastorTowerCollection posTowers, negTowers;
   
   // Call tower class
   Tower toweralgo;
   if (posCells.size() != 0) {
   	posTowers = toweralgo.runTowerProduction(posCells,5.9,towercut_);
	for (size_t i=0;i<posTowers.size();i++) {
		CastorTowers->push_back(posTowers[i]);
	}
   }
   if (negCells.size() != 0) {
   	negTowers = toweralgo.runTowerProduction(negCells,-5.9,towercut_);
	for (size_t i=0;i<negTowers.size();i++) {
		CastorTowers->push_back(negTowers[i]);
	}
   }
      
   iEvent.put(CastorTowers);
   
   } else {
   
   // Read the towers created by the Castor FastSimulation
   Handle<CastorTowerCollection> castorFastTowers;
   iEvent.getByLabel("CastorTowerReco", castorFastTowers);
   
   //CastorTowerCollection posTowers, negTowers;
   
   for (size_t i = 0; i < castorFastTowers->size(); ++i) {
   	// get properties
   	const CastorTower & ct = (*castorFastTowers)[i];
	if ( ct.eta() > 0.) {
		posTowers.push_back(ct);
	} else {
		negTowers.push_back(ct);
	}
   }
   
   
   }
   
   // Produce CastorJets from CastorTowers with Ktalgorithm
   KtAlgorithm ktalgo;
   auto_ptr<CastorJetCollection> CastorJetsfromKtAlgo (new CastorJetCollection);
   CastorJetCollection posJets;
   if (posTowers.size() != 0) {
   	posJets = ktalgo.runKtAlgo(posTowers,recom_,rParameter_);
   	for (size_t i=0; i<posJets.size(); i++) {
        	CastorJetsfromKtAlgo->push_back(posJets[i]);
   	}
   }
   CastorJetCollection negJets;
   if (negTowers.size() != 0) {
   	negJets = ktalgo.runKtAlgo(negTowers,recom_,rParameter_);
   	for (size_t i=0; i<negJets.size(); i++) {
        	CastorJetsfromKtAlgo->push_back(negJets[i]);
   	}
   }
   
   iEvent.put(CastorJetsfromKtAlgo, "fromKtAlgo");
   
   // Produce CastorEgammas from CastorJets
   Egamma egalgo;
   auto_ptr<CastorEgammaCollection> CastorEgammas (new CastorEgammaCollection);
   CastorEgammaCollection posEgammas, negEgammas;
   if (posJets.size() != 0) {
   	posEgammas = egalgo.runEgamma(posJets,minRatio_,maxWidth_,maxDepth_);
   	for (size_t i=0;i<posEgammas.size();i++) {
   		CastorEgammas->push_back(posEgammas[i]);
   	}
   }
   if (negJets.size() != 0) {
   	negEgammas = egalgo.runEgamma(negJets,minRatio_,maxWidth_,maxDepth_);
   	for (size_t i=0;i<negEgammas.size();i++) {
   		CastorEgammas->push_back(negEgammas[i]);
   	}
   }
   
   iEvent.put(CastorEgammas);
   
   
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
Castor::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
Castor::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(Castor);
