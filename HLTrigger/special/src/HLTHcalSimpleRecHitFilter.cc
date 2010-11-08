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
// $Id: HLTHcalSimpleRecHitFilter.cc,v 1.4 2010/08/26 00:47:05 frankma Exp $
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
    int minNHitsNeg_;
    int minNHitsPos_;
    bool doCoincidence_;
    std::vector<int> maskedList_;
    
};

//
// constructors and destructor
//
HLTHcalSimpleRecHitFilter::HLTHcalSimpleRecHitFilter(const edm::ParameterSet& iConfig)
{
    //now do what ever initialization is needed
    threshold_     = iConfig.getParameter<double>("threshold");
    minNHitsNeg_     = iConfig.getParameter<int>("minNHitsNeg");
    minNHitsPos_     = iConfig.getParameter<int>("minNHitsPos");
    doCoincidence_     = iConfig.getParameter<bool>("doCoincidence");
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
    
    bool accept = false ; 

    int nHitsNeg=0, nHitsPos=0;
    for ( HFRecHitCollection::const_iterator hitItr = crudeHits->begin(); hitItr != crudeHits->end(); ++hitItr ) {     
       HFRecHit hit = (*hitItr);
     
       // masking noisy channels
       std::vector<int>::iterator result;
       result = find( maskedList_.begin(), maskedList_.end(), HcalDetId(hit.id()).hashed_index() );    
       if  (result != maskedList_.end()) 
           continue; 
       
       // only count tower above threshold
       if ( hit.energy() < threshold_ ) continue;

       // count
       if (hit.id().zside()<0) ++nHitsNeg;
       else ++nHitsPos;
    }

    // Logic
    if (!doCoincidence_) accept = (nHitsNeg>=minNHitsNeg_) || (nHitsPos>=minNHitsPos_);
    else accept = (nHitsNeg>=minNHitsNeg_) && (nHitsPos>=minNHitsPos_);
//  edm::LogInfo("HcalFilter")  << "at evet: " << iEvent.id().event() 
//    << " and run: " << iEvent.id().run()
//    << " Total HF hits: " << crudeHits->size() << " Above Threshold - nNeg: " << nHitsNeg << " nPos " << nHitsPos
//    << " doCoinc: " << doCoincidence_ << " accept: " << accept << std::endl;

    // result
    return accept; 
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTHcalSimpleRecHitFilter);
