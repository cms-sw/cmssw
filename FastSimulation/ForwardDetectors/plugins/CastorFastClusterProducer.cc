// -*- C++ -*-
//
// Package:    CastorFastClusterProducer
// Class:      CastorFastClusterProducer
// 
/**\class CastorFastClusterProducer CastorFastClusterProducer.cc FastSimulation/ForwardDetectors/plugins/CastorFastClusterProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Hans Van Haevermaet
//         Created:  Thu Mar 13 12:00:56 CET 2008
// $Id: CastorFastClusterProducer.cc,v 1.3 2013/02/27 22:05:25 wdd Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <iostream>
#include <sstream>
#include <TMath.h>
#include <TRandom3.h>
#include <TF1.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/Point3D.h"

// Castorobject includes
#include "DataFormats/CastorReco/interface/CastorCluster.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"

// genCandidate particle includes
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "FastSimulation/ForwardDetectors/plugins/CastorFastClusterProducer.h"


//
// constructors and destructor
//
CastorFastClusterProducer::CastorFastClusterProducer(const edm::ParameterSet& iConfig) 
{
   //register your products
   produces<CastorClusterCollection>();
   
   //now do what ever other initialization is needed

}


CastorFastClusterProducer::~CastorFastClusterProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CastorFastClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   using namespace std;
   using namespace TMath;
   
   //
   // Make CastorCluster objects
   //
   
   //cout << "entering event" << endl;
   
   Handle<GenParticleCollection> genParticles;
   iEvent.getByLabel("genParticles", genParticles);
   
   // make pointer to towers that will be made
   auto_ptr<CastorClusterCollection> CastorClusters (new CastorClusterCollection);
   
   /*
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
	//castorplus[4][j] = 0.;
	//castormin[4][j] = 0.;
   }
   
   // declare properties vectors
   vector<double> depthplus[16];
   vector<double> depthmin[16];
   vector<double> fhotplus [16];
   vector<double> fhotmin [16];
   vector<double> energyplus [16];
   vector<double> energymin [16];
   
   for (int i=0;i<16;i++) {
   	depthplus[i].clear();
	depthmin[i].clear();
	fhotplus[i].clear();
	fhotmin[i].clear();
	energyplus[i].clear();
	energymin[i].clear();
   }
   */
   
   //cout << "declared everything" << endl;
   
   // start particle loop
   for (size_t i = 0; i < genParticles->size(); ++i) {
   	const Candidate & p = (*genParticles)[i];
	
	// select particles in castor
	if ( fabs(p.eta()) > 5.2 && fabs(p.eta()) < 6.6) {
	
	    //cout << "found particle in castor, start calculating" << endl;
	    
	    // declare energies
	    double gaus_E = -1.; 
   	    double emEnergy = 0.;
   	    double hadEnergy = 0.;
	    //double fhot = 0.;
	    //double depth = 0.;
	    
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
		
		// make cluster
		ClusterPoint pt1;
		if (p.eta() > 0.) {ClusterPoint temp(88.5,5.9,p.phi()); pt1 = temp;}
		if (p.eta() < 0.) {ClusterPoint temp(88.5,-5.9,p.phi()); pt1 = temp;}
   	    	Point pt2(pt1);
		CastorTowerRefVector refvector;
		CastorClusters->push_back(reco::CastorCluster(gaus_E,pt2,emEnergy,hadEnergy,emEnergy/gaus_E,0.,0.,0.,0.,refvector));
		
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
		
		// make cluster
		ClusterPoint pt1;
		if (p.eta() > 0.) {ClusterPoint temp(88.5,5.9,p.phi()); pt1 = temp;}
		if (p.eta() < 0.) {ClusterPoint temp(88.5,-5.9,p.phi()); pt1 = temp;}
   	    	Point pt2(pt1);
		CastorTowerRefVector refvector;
		CastorClusters->push_back(reco::CastorCluster(gaus_E,pt2,0.,hadEnergy,0.,0.,0.,0.,0.,refvector));
	    }
	    
	    /*
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
		
		depthplus[sector].push_back(depth);
		fhotplus[sector].push_back(fhot);
		energyplus[sector].push_back(gaus_E);
		//cout << "filled vectors" << endl;
		//cout << "energyplus size = " << energyplus[sector].size() << endl;
		//cout << "depthplus size = " << depthplus[sector].size() << endl;
		//cout << "fhotplus size = " << fhotplus[sector].size() << endl;
		
	    } else { 
		castormin[0][sector] = castormin[0][sector] + gaus_E;
		castormin[1][sector] = castormin[1][sector] + emEnergy;
		castormin[2][sector] = castormin[2][sector] + hadEnergy;
		
		
		depthmin[sector].push_back(depth);
		fhotmin[sector].push_back(fhot);
		energymin[sector].push_back(gaus_E);
		//cout << "filled vectors" << endl;
		
	    }
	    */
	    
	}
	
   }
   
   /*
   // substract pedestals/noise
   for (int j = 0; j < 16; j++) {
	double hadnoise = 0.;
	for (int i=0;i<12;i++) {
		hadnoise = hadnoise + make_noise();
	}
	castorplus[0][j] = castorplus[0][j] - hadnoise - make_noise() - make_noise();
	castormin[0][j] = castormin[0][j] - hadnoise - make_noise() - make_noise();
	castorplus[1][j] = castorplus[1][j] - make_noise() - make_noise();
	castormin[1][j] = castormin[1][j] - make_noise() - make_noise();
	castorplus[2][j] = castorplus[2][j] - hadnoise;
	castormin[2][j] = castormin[2][j] - hadnoise; 
	
	// set possible negative values to zero
	if (castorplus[0][j] < 0.) castorplus[0][j] = 0.;
	if (castormin[0][j] < 0.) castormin[0][j] = 0.;
	if (castorplus[1][j] < 0.) castorplus[1][j] = 0.;
	if (castormin[1][j] < 0.) castormin[1][j] = 0.;
	if (castorplus[2][j] < 0.) castorplus[2][j] = 0.;
	if (castormin[2][j] < 0.) castormin[2][j] = 0.;
   }
   */
   
   /*
   // store towers from castor arrays
   // eta = 5.9
   for (int j=0;j<16;j++) {
   	if (castorplus[0][j] > 0.) {
	    
	    double fem = 0.;
   	    fem = castorplus[1][j]/castorplus[0][j]; 
	    ClusterPoint pt1(88.5,5.9,castorplus[3][j]);
   	    Point pt2(pt1); 
	    
	    // parametrize depth and fhot from full sim
	    // get fit parameters from energy
	    // get random number according to distribution with fit parameters
	    double depth_mean = 0.;
	    double fhot_mean = 0.;
	    double sum_energy = 0.;
	    
	    //cout << "energyplus size = " << energyplus[j].size()<< endl;
	    for (size_t p = 0; p<energyplus[j].size();p++) {
	    	depth_mean = depth_mean + depthplus[j][p]*energyplus[j][p];
	    	fhot_mean = fhot_mean + fhotplus[j][p]*energyplus[j][p];
		sum_energy = sum_energy + energyplus[j][p];
	    }
	    depth_mean = depth_mean/sum_energy;
	    fhot_mean = fhot_mean/sum_energy;
	    cout << "computed depth/fhot" << endl;
	    
	    
	    edm::RefVector<edm::SortedCollection<CastorRecHit> > refvector;
	    CastorClusters->push_back(reco::CastorCluster(castorplus[0][j],pt2,castorplus[1][j],castorplus[2][j],fem,depth_mean,fhot_mean,refvector));	
	}
   }
   // eta = -5.9
   for (int j=0;j<16;j++) {
   	if (castormin[0][j] > 0.) {
	    double fem = 0.;
   	    fem = castormin[1][j]/castormin[0][j]; 
	    ClusterPoint pt1(88.5,-5.9,castormin[3][j]);
   	    Point pt2(pt1); 
	    
	    // parametrize depth and fhot from full sim
	    // get fit parameters from energy
	    // get random number according to distribution with fit parameters
	    double depth_mean = 0.;
	    double fhot_mean = 0.;
	    double sum_energy = 0.;
	    
	    
	    for (size_t p = 0; p<energymin[j].size();p++) {
	    	depth_mean = depth_mean + depthmin[j][p]*energymin[j][p];
	    	fhot_mean = fhot_mean + fhotmin[j][p]*energymin[j][p];
		sum_energy = sum_energy + energymin[j][p];
	    }
	    depth_mean = depth_mean/sum_energy;
	    fhot_mean = fhot_mean/sum_energy;
	    
	    
	    edm::RefVector<edm::SortedCollection<CastorRecHit> > refvector;
	    CastorClusters->push_back(reco::CastorCluster(castormin[0][j],pt2,castormin[1][j],castormin[2][j],fem,depth_mean,fhot_mean,refvector));	
	}
   }
   */
    	
   iEvent.put(CastorClusters); 
   
}

double CastorFastClusterProducer::make_noise() {
	double result = -1.;
	TRandom3 r2(0);
   	double mu_noise = 0.053; // GeV (from 1.214 ADC) per channel
   	double sigma_noise = 0.027; // GeV (from 0.6168 ADC) per channel
	
	while (result < 0.) {
		result = r2.Gaus(mu_noise,sigma_noise);
	}
	
	return result;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CastorFastClusterProducer);
