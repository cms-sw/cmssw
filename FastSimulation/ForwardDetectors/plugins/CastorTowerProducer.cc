// -*- C++ -*-
//
// Package:    CastorTowerProducer
// Class:      CastorTowerProducer
// 
/**\class CastorTowerProducer CastorTowerProducer.cc FastSimulation/CaloRecHitsProducer/plugins/CastorTowerProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Hans Van Haevermaet
//         Created:  Thu Mar 13 12:00:56 CET 2008
// $Id: CastorTowerProducer.cc,v 1.1 2008/11/30 15:57:20 beaudett Exp $
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
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorCell.h"

// genCandidate particle includes
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "FastSimulation/ForwardDetectors/plugins/CastorTowerProducer.h"


//
// constructors and destructor
//
CastorTowerProducer::CastorTowerProducer(const edm::ParameterSet& iConfig) 
{
   //register your products
   produces<CastorTowerCollection>();
   
   //now do what ever other initialization is needed

}


CastorTowerProducer::~CastorTowerProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CastorTowerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   using namespace std;
   using namespace TMath;
   
   //
   // Make CastorTower objects
   //
   
   Handle<GenParticleCollection> genParticles;
   iEvent.getByLabel("genParticles", genParticles);
   
   // make pointer to cluster that will be made
   auto_ptr<CastorTowerCollection> CastorTowers (new CastorTowerCollection);
   
   // declare castor array
   double castorplus [4][16]; // (0,x): Energies - (1,x): emEnergies - (2,x): hadEnergies - (3,x): phi position - eta = 5.9
   double castormin [4][16];  // (0,x): Energies - (1,x): emEnergies - (2,x): hadEnergies - (3,x): phi position - eta = -5.9
   // set phi values of array sectors and everything else to zero
   for (int j = 0; j < 16; j++) {
	castorplus[3][j] = -2.94524 + j*0.3927;
	castormin[3][j] = -2.94524 + j*0.3927;
	castorplus[0][j] = 0.;
	castormin[0][j] = 0.;
	castorplus[1][j] = 0.;
	castormin[1][j] = 0.;
	castorplus[2][j] = 0.;
	castormin[2][j] = 0.; 
   }
   
   // start particle loop
   for (size_t i = 0; i < genParticles->size(); ++i) {
   	const Candidate & p = (*genParticles)[i];
	
	// select particles in castor
	if ( fabs(p.eta()) > 5.2 && fabs(p.eta()) < 6.6) {
	    
	    // declare energies
	    double gaus_E = -1.; 
   	    double emEnergy = 0.;
   	    double hadEnergy = 0.;
	    
	    // add energies - em: if particle is e- or gamma
	    if (p.pdgId() == 11 || p.pdgId() == 22) {
	        
		while ( gaus_E < 0.) {
		// apply energy smearing with gaussian random generator
   	    	TRandom3 r(0);
   	    	// use sigma/E parametrization for the EM sections of CASTOR TB 2007 results
   	    	double sigma = p.energy()*(sqrt(pow(0.044,2) + pow(0.513/sqrt(p.energy()),2)));
   	    	gaus_E = r.Gaus(p.energy(),sigma);
		}
	    
	        // calculate electromagnetic electron/photon energy leakage
   	    	double tmax;
  	    	double a;
		double cte;
		if ( p.pdgId() == 11) { cte = -0.5; } else { cte = 0.5; }
   	    	tmax = 1.0*(log(gaus_E/0.0015)+cte);
   	    	a = tmax*0.5 + 1;
   	    	double leakage;
   	    	double x = 0.5*19.38;
   	    	leakage = gaus_E - gaus_E*Gamma(a,x);
		
		// add emEnergy
	    	emEnergy = gaus_E - leakage;
	    	// add hadEnergy leakage
		hadEnergy = leakage;
		
	    } else {
	    
	        while (gaus_E < 0.) {
	        // apply energy smearing with gaussian random generator
   	    	TRandom3 r(0);
   	    	// use sigma/E parametrization for the HAD sections of CASTOR TB 2007 results
		double sigma = p.energy()*(sqrt(pow(0.121,2) + pow(1.684/sqrt(p.energy()),2)));
   	    	gaus_E = r.Gaus(p.energy(),sigma);
		}
		
	        // add hadEnergy
		hadEnergy = gaus_E;

	    }
	    
	    // make tower
	    
	    // set sector
	    int sector = -1;
	    for (int j = 0; j < 16; j++) {
	        double a = -M_PI + j*0.3927;
		double b = -M_PI + (j+1)*0.3927;
	        if ( (p.phi() > a) && (p.phi() < b)) {  
		   sector = j;
		}
	    }
	    
	    // set eta
	    if (p.eta() > 0) { 
		castorplus[0][sector] = castorplus[0][sector] + gaus_E;
		castorplus[1][sector] = castorplus[1][sector] + emEnergy;
		castorplus[2][sector] = castorplus[2][sector] + hadEnergy;
	    } else { 
		castormin[0][sector] = castormin[0][sector] + gaus_E;
		castormin[1][sector] = castormin[1][sector] + emEnergy;
		castormin[2][sector] = castormin[2][sector] + hadEnergy;
	    }
	    
	}
	
   }
   
   // store towers from castor arrays
   // eta = 5.9
   for (int j=0;j<16;j++) {
   	if (castorplus[0][j] > 0.) {
	    double emtotRatio = 0.;
   	    emtotRatio = castorplus[1][j]/castorplus[0][j]; 
	    TowerPoint pt1(1.,5.9,castorplus[3][j]);
   	    Point pt2(pt1); 
	    std::vector<CastorCell> usedCells;
	    CastorTowers->push_back(reco::CastorTower(castorplus[0][j],pt2,castorplus[1][j],castorplus[2][j],emtotRatio,0.3927,0.,usedCells));
	    //posTowers.push_back(reco::CastorTower(castorplus[0][j],pt2,castorplus[1][j],castorplus[2][j],emtotRatio,0.3927,0.,usedCells));	
	}
   }
   // eta = -5.9
   for (int j=0;j<16;j++) {
   	if (castormin[0][j] > 0.) {
	    double emtotRatio = 0.;
   	    emtotRatio = castormin[1][j]/castormin[0][j]; 
	    TowerPoint pt1(1.,-5.9,castormin[3][j]);
   	    Point pt2(pt1); 
	    std::vector<CastorCell> usedCells;
	    CastorTowers->push_back(reco::CastorTower(castormin[0][j],pt2,castormin[1][j],castormin[2][j],emtotRatio,0.3927,0.,usedCells));
	    //negTowers.push_back(reco::CastorTower(castormin[0][j],pt2,castormin[1][j],castormin[2][j],emtotRatio,0.3927,0.,usedCells));	
	}
   }
    	
   iEvent.put(CastorTowers); 
   
}

// ------------ method called once each job just before starting event loop  ------------
void CastorTowerProducer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void CastorTowerProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(CastorTowerProducer);
