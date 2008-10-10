// -*- C++ -*-
//
// Package:    HLTFEDSizeFilter
// Class:      HLTFEDSizeFilter
// 
/**\class HLTFEDSizeFilter HLTFEDSizeFilter.cc Work/HLTFEDSizeFilter/src/HLTFEDSizeFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Bryan DAHMES
//         Created:  Wed Sep 19 16:21:29 CEST 2007
// $Id: HLTFEDSizeFilter.cc,v 1.2 2008/09/18 09:33:36 bdahmes Exp $
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

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

//
// class declaration
//

class HLTFEDSizeFilter : public HLTFilter {
public:
    explicit HLTFEDSizeFilter(const edm::ParameterSet&);
    ~HLTFEDSizeFilter();
    
private:
    virtual void beginJob(const edm::EventSetup&) ;
    virtual bool filter(edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;
    
    // ----------member data ---------------------------
    edm::InputTag RawCollection_;
    unsigned int threshold_;
    unsigned int fedStart_, fedStop_ ;
    
};

//
// constructors and destructor
//
HLTFEDSizeFilter::HLTFEDSizeFilter(const edm::ParameterSet& iConfig)
{
    //now do what ever initialization is needed
    threshold_  = iConfig.getParameter<unsigned int>("threshold");
    RawCollection_ = iConfig.getParameter<edm::InputTag>("rawData");
    // For a list of FEDs by subdetector, see DataFormats/FEDRawData/src/FEDNumbering.cc
    fedStart_   = iConfig.getParameter<unsigned int>("firstFED"); 
    fedStop_    = iConfig.getParameter<unsigned int>("lastFED");
}


HLTFEDSizeFilter::~HLTFEDSizeFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTFEDSizeFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

    // getting very basic uncalRH
    edm::Handle<FEDRawDataCollection> theRaw;
    try {
        iEvent.getByLabel(RawCollection_, theRaw);
    } catch ( std::exception& ex) {
        edm::LogWarning("HLTFEDSizeFilter") << RawCollection_ << " not available";
    }
    
    for (unsigned int i=fedStart_; i<=fedStop_; i++) 
        if ( theRaw->FEDData(i).size() > threshold_ ) return true ; 
    
    return false ; 
}

// ------------ method called once each job just before starting event loop  ------------
void 
HLTFEDSizeFilter::beginJob(const edm::EventSetup&) {
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTFEDSizeFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTFEDSizeFilter);
