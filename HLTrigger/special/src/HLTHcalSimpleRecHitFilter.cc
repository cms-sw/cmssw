// -*- C++ -*-
//
// Package:    HLTHcalSimpleRecHitFilter
// Class:      HLTHcalSimpleRecHitFilter
// 
/**\class HLTHcalSimpleRecHitFilter HLTHcalSimpleRecHitFilter.cc Work/HLTHcalSimpleRecHitFilter/src/HLTHcalSimpleRecHitFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Bryan DAHMES
//         Created:  Wed Sep 19 16:21:29 CEST 2007
// $Id: HLTHcalSimpleRecHitFilter.cc,v 1.2 2008/09/18 09:33:36 bdahmes Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

//
// class declaration
//

class HLTHcalSimpleRecHitFilter : public HLTFilter {
public:
    explicit HLTHcalSimpleRecHitFilter(const edm::ParameterSet&);
    ~HLTHcalSimpleRecHitFilter();
    
private:
    virtual bool filter(edm::Event&, const edm::EventSetup&);
    
    // ----------member data ---------------------------
    edm::InputTag HcalRecHitCollection_;
    double threshold_;
    std::vector<int> maskedList_;
    
};

//
// constructors and destructor
//
HLTHcalSimpleRecHitFilter::HLTHcalSimpleRecHitFilter(const edm::ParameterSet& iConfig)
{
    //now do what ever initialization is needed
    threshold_     = iConfig.getParameter<double>("threshold");
    maskedList_    = iConfig.getParameter<std::vector<int> >("maskedChannels"); //this is using the hashed index
    HcalRecHitCollection_ = iConfig.getParameter<edm::InputTag>("HFRecHitCollection");
    
}


HLTHcalSimpleRecHitFilter::~HLTHcalSimpleRecHitFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTHcalSimpleRecHitFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // using namespace edm;

    // getting very basic uncalRH
    edm::Handle<HFRecHitCollection> crudeHits;
    try {
        iEvent.getByLabel(HcalRecHitCollection_, crudeHits);
    } catch ( std::exception& ex) {
        edm::LogWarning("HLTHcalSimpleRecHitFilter") << HcalRecHitCollection_ << " not available";
    }
    
    bool aboveThreshold = false ; 

    for ( HFRecHitCollection::const_iterator hitItr = crudeHits->begin(); hitItr != crudeHits->end(); ++hitItr ) {     
       HFRecHit hit = (*hitItr);
     
       // masking noisy channels
       std::vector<int>::iterator result;
       result = find( maskedList_.begin(), maskedList_.end(), HcalDetId(hit.id()).hashed_index() );    
       if  (result != maskedList_.end()) 
           continue; 
       
       float ampli_ = hit.energy();
       
       if ( ampli_ >= threshold_ ) {
           aboveThreshold = true;
//            edm::LogInfo("HcalFilter")  << "at evet: " << iEvent.id().event() 
//                                           << " and run: " << iEvent.id().run() 
//                                           << " signal above threshold found: " << ampli_ ; 
           break ;
       }
    }
    return aboveThreshold ; 
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTHcalSimpleRecHitFilter);
