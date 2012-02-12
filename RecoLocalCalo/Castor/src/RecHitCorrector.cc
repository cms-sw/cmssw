// -*- C++ -*-
//
// Package:    RecHitCorrector
// Class:      RecHitCorrector
// 
/**\class RecHitCorrector RecHitCorrector.cc RecoLocalCalo/Castor/src/RecHitCorrector.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Hans Van Haevermaet
//         Created:  Wed Feb 23 11:29:43 CET 2011
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"


//
// class declaration
//

class RecHitCorrector : public edm::EDProducer {
   public:
      explicit RecHitCorrector(const edm::ParameterSet&);
      ~RecHitCorrector();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag inputLabel_;
      double factor_;
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
RecHitCorrector::RecHitCorrector(const edm::ParameterSet& iConfig):
inputLabel_(iConfig.getParameter<edm::InputTag>("rechitLabel")),
factor_(iConfig.getParameter<double>("revertFactor"))
{
   //register your products
   produces<CastorRecHitCollection>();
   //now do what ever other initialization is needed
  
}


RecHitCorrector::~RecHitCorrector()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
RecHitCorrector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   edm::Handle<CastorRecHitCollection> rechits;
   iEvent.getByLabel(inputLabel_,rechits);
   
   // get conditions
   edm::ESHandle<CastorDbService> conditions;
   iSetup.get<CastorDbRecord>().get(conditions);
   
   if (!rechits.isValid()) std::cout << "No valid CastorRecHitCollection found, please check the InputLabel..." << std::endl;
   
   CastorCalibrations calibrations;
   
   std::auto_ptr<CastorRecHitCollection> rec(new CastorRecHitCollection);
   
   for (unsigned int i=0;i<rechits->size();i++) {
   	CastorRecHit rechit = (*rechits)[i];
	//std::cout << "rechit energy = " << rechit.energy() << std::endl;
	double fC = factor_*rechit.energy();
	double time = rechit.time();
	//std::cout << "rechit energy(fC) = " << fC << " time = " << time << std::endl;
	
	// do proper gain calibration reading the latest entries in the condDB
	const CastorCalibrations& calibrations=conditions->getCastorCalibrations(rechit.id());
	int capid = 0; // take some capid
	double correctedenergy = fC*calibrations.gain(capid);
	//std::cout << " correctedenergy = " << correctedenergy << " gain = " << calibrations.gain(capid) << std::endl;
	
	CastorRecHit *correctedhit = new CastorRecHit(rechit.id(),correctedenergy,time);
	rec->push_back(*correctedhit);
   }
   
   iEvent.put(rec);
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
RecHitCorrector::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RecHitCorrector::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(RecHitCorrector);
