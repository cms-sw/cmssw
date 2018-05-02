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
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

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
#include "CondFormats/CastorObjects/interface/CastorChannelQuality.h"
#include "CondFormats/CastorObjects/interface/CastorChannelStatus.h"
#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"

//
// class declaration
//

class RecHitCorrector : public edm::stream::EDProducer<> {
   public:
      explicit RecHitCorrector(const edm::ParameterSet&);
      ~RecHitCorrector() override;

   private:
      void produce(edm::Event&, const edm::EventSetup&) override;
      
      // ----------member data ---------------------------
      edm::EDGetTokenT<CastorRecHitCollection> tok_input_;
      double factor_;
      bool doInterCalib_;
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
factor_(iConfig.getParameter<double>("revertFactor")),
doInterCalib_(iConfig.getParameter<bool>("doInterCalib"))
{
  tok_input_ = consumes<CastorRecHitCollection>(iConfig.getParameter<edm::InputTag>("rechitLabel"));
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
   
   // get original rechits
   edm::Handle<CastorRecHitCollection> rechits;
   iEvent.getByToken(tok_input_,rechits);
   
   // get conditions
   edm::ESHandle<CastorDbService> conditions;
   iSetup.get<CastorDbRecord>().get(conditions);
   
   edm::ESHandle<CastorChannelQuality> p;
   iSetup.get<CastorChannelQualityRcd>().get(p);
   CastorChannelQuality* myqual = new CastorChannelQuality(*p.product());
   
   if (!rechits.isValid()) edm::LogWarning("CastorRecHitCorrector") << "No valid CastorRecHitCollection found, please check the InputLabel...";
   
   CastorCalibrations calibrations;
   
   auto rec = std::make_unique<CastorRecHitCollection>();
   
   for (unsigned int i=0;i<rechits->size();i++) {
   	CastorRecHit rechit = (*rechits)[i];
	double time = rechit.time();
	double correctedenergy = factor_*rechit.energy();
	
	if (doInterCalib_) {
		// do proper gain calibration reading the latest entries in the condDB
		const CastorCalibrations& calibrations=conditions->getCastorCalibrations(rechit.id());
		int capid = 0; // take some capid, gains are the same for all capid's
		correctedenergy *= calibrations.gain(capid);
	}
	
	// now check the channelquality of this rechit
	bool ok = true;
	DetId detcell=(DetId)rechit.id();
	std::vector<DetId> channels = myqual->getAllChannels();
	for (auto channel : channels) {	
		if (channel.rawId() == detcell.rawId()) {
			const CastorChannelStatus* mydigistatus=myqual->getValues(channel);
			if (mydigistatus->getValue() == 2989) {
				ok = false; // 2989 = BAD
				break;
			}
		}
	}
	
	if (ok) {
	    rec->emplace_back(rechit.id(),correctedenergy,time);
	}
   }
   
   iEvent.put(std::move(rec));
   
   delete myqual;
 
}


//define this as a plug-in
DEFINE_FWK_MODULE(RecHitCorrector);
