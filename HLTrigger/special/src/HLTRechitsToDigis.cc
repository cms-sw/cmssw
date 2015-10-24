//
// Package:    HLTrigger/special
// Class:      HLTRechitsToDigis
// 
/**\class HLTRechitsToDigis HLTRechitsToDigis.cc HLTrigger/special/plugins/HLTRechitsToDigis.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Joshua Robert Hardenbrook
//         Created:  Fri, 20 Feb 2015 15:51:36 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

//
// class declaration
//

class HLTRechitsToDigis : public edm::stream::EDProducer<> {
public:
  explicit HLTRechitsToDigis(const edm::ParameterSet&);
  ~HLTRechitsToDigis();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  enum ecalRegion {invalidRegion=0, barrel, endcap };  
  static const HLTRechitsToDigis::ecalRegion stringToRegion(const std::string& region);

private:
  virtual void produce( edm::Event&, edm::EventSetup const&) override;
      
  // ----------member data ---------------------------  
  // tokens for the digi and rechits for matching
  edm::EDGetTokenT<EBDigiCollection> digisEBInToken_;
  edm::EDGetTokenT<EEDigiCollection> digisEEInToken_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitsToken_;
  
  // input tags
  edm::InputTag digisIn_;
  edm::InputTag recHits_;
  
  // string for the produced digi collection
  std::string digisOut_;
  ecalRegion region_;
};
//

//
// constructors and destructor
//
HLTRechitsToDigis::HLTRechitsToDigis(const edm::ParameterSet& iConfig)
{
  //region to do rechit digi matching
  region_ = stringToRegion(iConfig.getParameter<std::string> ("region"));

  // digis to match to hit collections
  digisIn_ = iConfig.getParameter<edm::InputTag> ("digisIn");
  digisOut_ = iConfig.getParameter<std::string> ("digisOut");

  // hit collections to save digis for
  recHits_ = iConfig.getParameter<edm::InputTag> ("recHits");

  // region specific tokens
  switch(region_) {
  case barrel:
    digisEBInToken_ = consumes<EBDigiCollection>(digisIn_);
    produces<EBDigiCollection>(digisOut_);  
    break;
  case endcap:
    digisEEInToken_ = consumes<EEDigiCollection>(digisIn_);
    produces<EEDigiCollection>(digisOut_);  
    break;    
  case invalidRegion:  
    break;
  }

  recHitsToken_ = consumes<EcalRecHitCollection>(recHits_);
}

HLTRechitsToDigis::~HLTRechitsToDigis()
{ 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//
const HLTRechitsToDigis::ecalRegion
HLTRechitsToDigis::stringToRegion(const std::string & region) {  
  if (region == "barrel")     return barrel;
  else if (region == "endcap")     return endcap;
  else return invalidRegion;
}


// ------------ method called to produce the data  ------------
void 
HLTRechitsToDigis::produce(edm::Event& iEvent, edm::EventSetup const& setup)  {
  using namespace edm;
  // handles for digis
  Handle<EBDigiCollection> digisEBHandle;
  Handle<EEDigiCollection> digisEEHandle;
  
  // output collections
  std::auto_ptr<EBDigiCollection> outputEBDigiCollection( new EBDigiCollection );
  std::auto_ptr<EEDigiCollection> outputEEDigiCollection( new EEDigiCollection );
  
  // calibrated rechits
  Handle<EcalRecHitCollection> recHitsHandle;
  iEvent.getByToken(recHitsToken_, recHitsHandle);   
  
  // match the digis based on the region
  switch(region_) {
  case barrel: { 
    iEvent.getByToken(digisEBInToken_, digisEBHandle);
    const EBDigiCollection* digisEB = digisEBHandle.product();   
    
    // loop over the collection of rechits and match to digis
    EcalRecHitCollection::const_iterator ituneEB;
    for (ituneEB = recHitsHandle->begin(); ituneEB != recHitsHandle->end(); ituneEB++) {
      EcalRecHit const &    hit = (*ituneEB);
      EcalDigiCollection::const_iterator digiLookUp = digisEB->find(hit.id());
      // protect against a digi not existing
      if( digiLookUp == digisEB->end()) continue;
      outputEBDigiCollection->push_back( digiLookUp->id(), digiLookUp->begin() );
    }

    // add the built collection to the event 
    iEvent.put( outputEBDigiCollection, digisOut_);     
    break;      
  }    
  case endcap: {
    iEvent.getByToken(digisEEInToken_, digisEEHandle);
    const EEDigiCollection* digisEE = digisEEHandle.product();   
    
    // loop over the collection of rechits and match to digis
    EcalRecHitCollection::const_iterator ituneEE;
    for (ituneEE = recHitsHandle->begin(); ituneEE != recHitsHandle->end(); ituneEE++) {
      EcalRecHit const & hit = (*ituneEE);            
      EcalDigiCollection::const_iterator digiLookUp = digisEE->find(hit.id());
      // protect against a digi not existing for the saved rechit
      if(digiLookUp  == digisEE->end()) continue;
      outputEEDigiCollection->push_back( digiLookUp->id(), digiLookUp->begin() );              
    } // end loop over endcap rechits

    // add the built collection to the event     
    iEvent.put(outputEEDigiCollection, digisOut_);     
    break;
  }
  case invalidRegion: {
    break;
  }
  } // end switch statement for the region (barrel, endcap, invalid)
}

// ------------ method called when starting to processes a run  ------------
/*
void
HLTRechitsToDigis::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
HLTRechitsToDigis::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
HLTRechitsToDigis::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
HLTRechitsToDigis::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HLTRechitsToDigis::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;

  desc.add<std::string>("region", "barrel")
    ->setComment("Region of rechits to save Digis for. Allowed values: barrel or endcap.");
  desc.add<edm::InputTag>("digisIn",edm::InputTag("ecalDigis","ebDigis"))
    ->setComment("The collection of either barrel or endcap digis which correspond to the rechit collection");
  desc.add<std::string>("digisOut","pi0EBDigis")
    ->setComment("Name for the collection of Digis saved by the module");
  desc.add<edm::InputTag>("recHits",edm::InputTag("hltAlCaPi0EBUncalibrator","pi0EcalRecHitsEB"))
    ->setComment("Collection of rechits to match Digis to");  
  descriptions.add("hltFindMatchingECALDigisToRechits", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTRechitsToDigis);
