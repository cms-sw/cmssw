// -*- C++ -*-
//
// Package:    CastorFastTowerProducer
// Class:      CastorFastTowerProducer
// 
/**\class CastorFastTowerProducer CastorFastTowerProducer.cc FastSimulation/ForwardDetectors/plugins/CastorFastTowerProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Hans Van Haevermaet
//         Created:  Thu Mar 13 12:00:56 CET 2008
// $Id: CastorFastTowerProducer.cc,v 1.4 2013/02/27 22:05:25 wdd Exp $
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
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"

// genCandidate particle includes
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "FastSimulation/ForwardDetectors/plugins/CastorFastTowerProducer.h"


//
// constructors and destructor
//
CastorFastTowerProducer::CastorFastTowerProducer(const edm::ParameterSet& iConfig) 
{
   //register your products
   produces<CastorTowerCollection>();
   
   //now do what ever other initialization is needed

}


CastorFastTowerProducer::~CastorFastTowerProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CastorFastTowerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   using namespace std;
   using namespace TMath;
   
   //
   // Make CastorTower objects
   //
   
   //cout << "entering event" << endl;
   
   Handle<GenParticleCollection> genParticles;
   iEvent.getByLabel("genParticles", genParticles);
   
   // make pointer to towers that will be made
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
   //vector<double> femplus [16];
   //vector<double> femmin [16];
   
   for (int i=0;i<16;i++) {
   	depthplus[i].clear();
	depthmin[i].clear();
	fhotplus[i].clear();
	fhotmin[i].clear();
	energyplus[i].clear();
	energymin[i].clear();
	//femplus[i].clear();
	//femmin[i].clear();
   }
   
   //cout << "declared everything" << endl;
   
   // start particle loop
   for (size_t i = 0; i < genParticles->size(); ++i) {
   	const Candidate & p = (*genParticles)[i];
	
	// select particles in castor
	if ( fabs(p.eta()) > 5.2 && fabs(p.eta()) < 6.6) {
	
	    //cout << "found particle in castor, start calculating" << endl;
	    
	    // declare energies
	    //double gaus_E = -1.; 
	    double energy = -1.;
   	    double emEnergy = 0.;
   	    double hadEnergy = 0.;
	    double fhot = 0.;
	    double depth = 0.;
	    //double fem = 0.;
	    
	    // add energies - em: if particle is e- or gamma
	    if (p.pdgId() == 11 || p.pdgId() == 22) {
	        
		// calculate primary tower energy for electrons
		while ( energy < 0.) {
		// apply energy smearing with gaussian random generator
   	    	TRandom3 r(0);
   	    	// use sigma/E parametrization from the full simulation
		double mean = 1.0024*p.energy() - 0.3859;
   	    	double sigma = 0.0228*p.energy() + 2.1061;
   	    	energy = r.Gaus(mean,sigma);
		}
	    
	        // calculate electromagnetic electron/photon energy leakage
   	    	double tmax;
  	    	double a;
		double cte;
		if ( p.pdgId() == 11) { cte = -0.5; } else { cte = 0.5; }
   	    	tmax = 1.0*(log(energy/0.0015)+cte);
   	    	a = tmax*0.5 + 1;
   	    	double leakage;
   	    	double x = 0.5*19.38;
   	    	leakage = energy - energy*Gamma(a,x);
		
		// add emEnergy
	    	emEnergy = energy - leakage;
	    	// add hadEnergy leakage
		hadEnergy = leakage;
		
		// calculate EM depth from parametrization
		double d0 = 0.2338 * pow(p.energy(),-0.1634);
		double d1 = 5.4336 * pow(p.energy(),0.2410) + 14408.1025;
		double d2 = 1.4692 * pow(p.energy(),0.1307) - 0.5216; 
		if (d0 < 0.) d0 = abs(d0);
		
		TF1 *fdepth = new TF1("fdepth","[0] * TMath::Exp(-0.5*( (x-[1])/[2] + TMath::Exp(-(x-[1])/[2])))",14400.,14460.); 
		fdepth->SetParameters(d0,d1,d2);
		depth = fdepth->GetRandom();
		fdepth->Delete();
		if (p.eta() < 0.) depth = -1*depth;
		
	    } else {
	    	
		// calculate primary tower energy for hadrons
	        while (energy < 0.) {
	        // apply energy smearing with gaussian random generator
   	    	TRandom3 r(0);
   	    	// use sigma/E parametrization from the full simulation
		double mean = 0.8340*p.energy() - 8.5054;
		double sigma = 0.1595*p.energy() + 3.1183;
   	    	energy = r.Gaus(mean,sigma);
		}
		
	        // add hadEnergy
		hadEnergy = energy;
		
		// in the near future add fem parametrization
		
		// calculate depth for HAD particle from parametrization
		double d0 = -0.000012 * p.energy() + 0.0661;
		double d1 = 785.7524 * pow(p.energy(),0.0262) + 13663.4262;
		double d2 = 9.8748 * pow(p.energy(),0.1720) + 37.0187; 
		if (d0 < 0.) d0 = abs(d0);
		
		TF1 *fdepth = new TF1("fdepth","[0] * TMath::Exp(-0.5*( (x-[1])/[2] + TMath::Exp(-(x-[1])/[2]) ))",14400.,15500.);
		fdepth->SetParameters(d0,d1,d2);
   		depth = fdepth->GetRandom();
		fdepth->Delete();
		if (p.eta() < 0.) depth = -1*depth;
		

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
		castorplus[0][sector] = castorplus[0][sector] + energy;
		castorplus[1][sector] = castorplus[1][sector] + emEnergy;
		castorplus[2][sector] = castorplus[2][sector] + hadEnergy;
		
		depthplus[sector].push_back(depth);
		fhotplus[sector].push_back(fhot);
		energyplus[sector].push_back(energy);
		//cout << "filled vectors" << endl;
		//cout << "energyplus size = " << energyplus[sector].size() << endl;
		//cout << "depthplus size = " << depthplus[sector].size() << endl;
		//cout << "fhotplus size = " << fhotplus[sector].size() << endl;
		
	    } else { 
		castormin[0][sector] = castormin[0][sector] + energy;
		castormin[1][sector] = castormin[1][sector] + emEnergy;
		castormin[2][sector] = castormin[2][sector] + hadEnergy;
		
		
		depthmin[sector].push_back(depth);
		fhotmin[sector].push_back(fhot);
		energymin[sector].push_back(energy);
		//cout << "filled vectors" << endl;
		
	    }
	    
	}
	
   }
   
   
   // add and substract pedestals/noise
   for (int j = 0; j < 16; j++) {
	double hadnoise = 0.;
	double emnoise = 0.;
	for (int i=0;i<12;i++) {
		hadnoise = hadnoise + make_noise();
		if (i<2) emnoise = emnoise + make_noise();
	}
	
	hadnoise = hadnoise - 12*0.053;
	emnoise = emnoise - 2*0.053;
	if ( hadnoise < 0.) hadnoise = 0.;
	if ( emnoise < 0.) emnoise = 0.;
	double totnoise = hadnoise + emnoise;
	
	// add random noise
	castorplus[0][j] = castorplus[0][j] + totnoise;
	castormin[0][j] = castormin[0][j] + totnoise;
	castorplus[1][j] = castorplus[1][j] + emnoise;
	castormin[1][j] = castormin[1][j] + emnoise;
	castorplus[2][j] = castorplus[2][j] + hadnoise;
	castormin[2][j] = castormin[2][j] + hadnoise; 

	//cout << "after constant substraction" << endl;
	//cout << "total noise = " << castorplus[0][j] << " em noise = " << castorplus[1][j] << " had noise = " << castorplus[2][j] << endl;
	//cout << "fem should be = " << castorplus[1][j]/castorplus[0][j] << endl;
	
   }
   
   
   // store towers from castor arrays
   // eta = 5.9
   for (int j=0;j<16;j++) {
   	if (castorplus[0][j] != 0.) {
	    
	    double fem = 0.;
   	    fem = castorplus[1][j]/castorplus[0][j]; 
	    TowerPoint pt1(88.5,5.9,castorplus[3][j]);
   	    Point pt2(pt1); 
	    
	    //cout << "fem became = " << castorplus[1][j]/castorplus[0][j] << endl;
	    
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
	    //cout << "computed depth/fhot" << endl;
	    
	    
	    //std::vector<CastorCell> usedCells;
	    edm::RefVector<edm::SortedCollection<CastorRecHit> > refvector;
	    CastorTowers->push_back(reco::CastorTower(castorplus[0][j],pt2,castorplus[1][j],castorplus[2][j],fem,depth_mean,fhot_mean,refvector));	
	}
   }
   // eta = -5.9
   for (int j=0;j<16;j++) {
   	if (castormin[0][j] != 0.) {
	    double fem = 0.;
   	    fem = castormin[1][j]/castormin[0][j]; 
	    TowerPoint pt1(88.5,-5.9,castormin[3][j]);
   	    Point pt2(pt1); 
	    
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
	    
	    
	    //std::vector<CastorCell> usedCells;
	    edm::RefVector<edm::SortedCollection<CastorRecHit> > refvector;
	    CastorTowers->push_back(reco::CastorTower(castormin[0][j],pt2,castormin[1][j],castormin[2][j],fem,depth_mean,fhot_mean,refvector));	
	}
   }
    	
   iEvent.put(CastorTowers); 
   
}

double CastorFastTowerProducer::make_noise() {
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
DEFINE_FWK_MODULE(CastorFastTowerProducer);
