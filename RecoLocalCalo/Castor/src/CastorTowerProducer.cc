// -*- C++ -*-
//
// Package:    Castor
// Class:      CastorTowerProducer
// 

/**\class CastorTowerProducer CastorTowerProducer.cc RecoLocalCalo/Castor/src/CastorTowerProducer.cc

 Description: CastorTower Reconstruction Producer. Produce CastorTowers from CastorCells.
 Implementation:
*/

//
// Original Author:  Hans Van Haevermaet, Benoit Roland
//         Created:  Wed Jul  9 14:00:40 CEST 2008
// $Id: CastorTowerProducer.cc,v 1.12 2013/01/07 15:32:45 hvanhaev Exp $
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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/Point3D.h"

// Castor Object include
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"

// Channel quality
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include "CondFormats/CastorObjects/interface/CastorChannelStatus.h"
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"

//
// class declaration
//

class CastorTowerProducer : public edm::EDProducer {
   public:
      explicit CastorTowerProducer(const edm::ParameterSet&);
      ~CastorTowerProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void ComputeTowerVariable(const edm::RefVector<edm::SortedCollection<CastorRecHit> >& usedRecHits, double&  Ehot, double& depth);
      
      // member data
      typedef math::XYZPointD Point;
      typedef ROOT::Math::RhoEtaPhiPoint TowerPoint;
      typedef ROOT::Math::RhoZPhiPoint CellPoint;
      typedef edm::SortedCollection<CastorRecHit> CastorRecHitCollection; 
      typedef std::vector<reco::CastorTower> CastorTowerCollection;
      typedef edm::RefVector<CastorRecHitCollection> CastorRecHitRefVector;
      std::string input_;
      double towercut_;
      double mintime_;
      double maxtime_;
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

CastorTowerProducer::CastorTowerProducer(const edm::ParameterSet& iConfig) :
  input_(iConfig.getParameter<std::string>("inputprocess")),
  towercut_(iConfig.getParameter<double>("towercut")),
  mintime_(iConfig.getParameter<double>("mintime")),
  maxtime_(iConfig.getParameter<double>("maxtime"))
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
void CastorTowerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  using namespace reco;
  using namespace TMath;
  
  // Produce CastorTowers from CastorCells
  
  edm::Handle<CastorRecHitCollection> InputRecHits;
  iEvent.getByLabel(input_,InputRecHits);

  std::auto_ptr<CastorTowerCollection> OutputTowers (new CastorTowerCollection);
   
  // get and check input size
  int nRecHits = InputRecHits->size();

  LogDebug("CastorTowerProducer")
    <<"2. entering CastorTowerProducer"<<std::endl;

  if (nRecHits==0)
    LogDebug("CastorTowerProducer") <<"Warning: You are trying to run the Tower algorithm with 0 input rechits.";
  
  // declare castor array
  // (0,x): Energies - (1,x): emEnergies - (2,x): hadEnergies - (3,x): phi position
  
  double poscastortowerarray[4][16]; 
  double negcastortowerarray[4][16];

  CastorRecHitRefVector poscastorusedrechits[16];
  CastorRecHitRefVector negcastorusedrechits[16];

  // set phi values and everything else to zero
  for (int j = 0; j < 16; j++) {
    poscastortowerarray[3][j] = -2.94524 + j*0.3927;
    poscastortowerarray[0][j] = 0.;
    poscastortowerarray[1][j] = 0.;
    poscastortowerarray[2][j] = 0.;

    negcastortowerarray[3][j] = -2.94524 + j*0.3927;
    negcastortowerarray[0][j] = 0.;
    negcastortowerarray[1][j] = 0.;
    negcastortowerarray[2][j] = 0.;
  }
  
  // retrieve the channel quality lists from database
  edm::ESHandle<CastorChannelQuality> p;
  iSetup.get<CastorChannelQualityRcd>().get(p);
  std::vector<DetId> channels = p->getAllChannels();

  // loop over rechits to build castortowerarray[4][16] and castorusedrechits[16] 
  for (unsigned int i = 0; i < InputRecHits->size(); i++) {
    
    edm::Ref<CastorRecHitCollection> rechit_p = edm::Ref<CastorRecHitCollection>(InputRecHits, i);
    
    HcalCastorDetId id = rechit_p->id();
    DetId genericID=(DetId)id;
    
    // first check if the rechit is in the BAD channel list
    bool bad = false;
    for (std::vector<DetId>::iterator channel = channels.begin();channel !=  channels.end();channel++) {	
    	if (channel->rawId() == genericID.rawId()) {
		// if the rechit is found in the list, set it bad
	  bad = true; break;
    	}
    }
    // if bad, continue the loop to the next rechit
    if (bad) continue;
    
    double Erechit = rechit_p->energy();
    int module = id.module();
    int sector = id.sector();
    double zrechit = 0;
    if (module < 3) zrechit = -14390 - 24.75 - 49.5*(module-1);
    if (module > 2) zrechit = -14390 - 99 - 49.5 - 99*(module-3); 
    double phirechit = -100;
    if (sector < 9) phirechit = 0.19635 + (sector-1)*0.3927;
    if (sector > 8) phirechit = -2.94524 + (sector - 9)*0.3927;

    // add time conditions for the rechit
    if (rechit_p->time() > mintime_ && rechit_p->time() < maxtime_) {

    	// loop over the 16 towers possibilities
    	for ( int j=0;j<16;j++) {
      
      		// phi matching condition
      		if (TMath::Abs(phirechit - poscastortowerarray[3][j]) < 0.0001) {

			// condition over rechit z value
    			if (zrechit > 0.) {
	  			poscastortowerarray[0][j]+=Erechit;
	  			if (module < 3) {poscastortowerarray[1][j]+=Erechit;} else {poscastortowerarray[2][j]+=Erechit;}
	  			poscastorusedrechits[j].push_back(rechit_p);
			} else {
	  			negcastortowerarray[0][j]+=Erechit;
	  			if (module < 3) {negcastortowerarray[1][j]+=Erechit;} else {negcastortowerarray[2][j]+=Erechit;}
	  			negcastorusedrechits[j].push_back(rechit_p);
			} // end condition over rechit z value
      		} // end phi matching condition
    	} // end loop over the 16 towers possibilities
    } // end time conditions
    
  } // end loop over rechits to build castortowerarray[4][16] and castorusedrechits[16]
  
  // make towers of the arrays

  double fem, Ehot, depth;
  double rhoTower = 88.5;

  // loop over the 16 towers possibilities
  for (int k=0;k<16;k++) {
    
    fem = 0;
    Ehot = 0;
    depth = 0;

    // select the positive towers with E > sqrt(Nusedrechits)*Ecut
    if (poscastortowerarray[0][k] > sqrt(poscastorusedrechits[k].size())*towercut_) {
      
      fem = poscastortowerarray[1][k]/poscastortowerarray[0][k];
      CastorRecHitRefVector usedRecHits = poscastorusedrechits[k];
      ComputeTowerVariable(usedRecHits,Ehot,depth);

      LogDebug("CastorTowerProducer")
	<<"tower "<<k+1<<": fem = "<<fem<<" ,depth = "<<depth<<" ,Ehot = "<<Ehot<<std::endl;

      TowerPoint temptowerposition(rhoTower,5.9,poscastortowerarray[3][k]);
      Point towerposition(temptowerposition);

      CastorTower newtower(poscastortowerarray[0][k],towerposition,poscastortowerarray[1][k],poscastortowerarray[2][k],fem,depth,Ehot,
			   poscastorusedrechits[k]);
      OutputTowers->push_back(newtower);
    } // end select the positive towers with E > Ecut
    
    // select the negative towers with E > sqrt(Nusedrechits)*Ecut
    if (negcastortowerarray[0][k] > sqrt(negcastorusedrechits[k].size())*towercut_) {
      
      fem = negcastortowerarray[1][k]/negcastortowerarray[0][k];
      CastorRecHitRefVector usedRecHits = negcastorusedrechits[k];
      ComputeTowerVariable(usedRecHits,Ehot,depth);

      LogDebug("CastorTowerProducer")
	 <<"tower "<<k+1 << " energy = " << negcastortowerarray[0][k] << "EM = " << negcastortowerarray[1][k] << "HAD = " << negcastortowerarray[2][k] << "phi = " << negcastortowerarray[3][k] << ": fem = "<<fem<<" ,depth = "<<depth<<" ,Ehot = "<<Ehot<<std::endl;

      TowerPoint temptowerposition(rhoTower,-5.9,negcastortowerarray[3][k]);
      Point towerposition(temptowerposition);

      CastorTower newtower(negcastortowerarray[0][k],towerposition,negcastortowerarray[1][k],negcastortowerarray[2][k],fem,depth,Ehot,
			   negcastorusedrechits[k]);
      OutputTowers->push_back(newtower);
    } // end select the negative towers with E > Ecut
    
  } // end loop over the 16 towers possibilities
  
  iEvent.put(OutputTowers);
} 


// ------------ method called once each job just before starting event loop  ------------
void CastorTowerProducer::beginJob() {
  LogDebug("CastorTowerProducer")
    <<"Starting CastorTowerProducer";
}

// ------------ method called once each job just after ending the event loop  ------------
void CastorTowerProducer::endJob() {
  LogDebug("CastorTowerProducer")
    <<"Ending CastorTowerProducer";
}

void CastorTowerProducer::ComputeTowerVariable(const edm::RefVector<edm::SortedCollection<CastorRecHit> >& usedRecHits, double&  Ehot, double& depth) {

  using namespace reco;

  double Etot = 0;

  // loop over the cells used in the tower k
  for (CastorRecHitRefVector::iterator it = usedRecHits.begin(); it != usedRecHits.end(); it++) {
    edm::Ref<CastorRecHitCollection> rechit_p = *it;

    double Erechit = rechit_p->energy();
    HcalCastorDetId id = rechit_p->id();
    int module = id.module();
    double zrechit = 0;
    if (module < 3) zrechit = -14390 - 24.75 - 49.5*(module-1);
    if (module > 2) zrechit = -14390 - 99 - 49.5 - 99*(module-3); 

    if(Erechit > Ehot) Ehot = Erechit;
    depth+=Erechit*zrechit;
    Etot+=Erechit;
  }

  depth/=Etot;
  Ehot/=Etot;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CastorTowerProducer);
