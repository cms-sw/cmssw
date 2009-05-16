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
// $Id: HLTFEDSizeFilter.cc,v 1.6 2009/05/16 12:01:09 fwyzard Exp $
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
    virtual bool filter(edm::Event&, const edm::EventSetup&);
    
    // ----------member data ---------------------------
    edm::InputTag RawCollection_;
    unsigned int  threshold_;
    unsigned int  fedStart_, fedStop_ ;
    bool          requireAllFEDs_;
 
};

//
// constructors and destructor
//
HLTFEDSizeFilter::HLTFEDSizeFilter(const edm::ParameterSet& iConfig)
{
    threshold_      = iConfig.getParameter<unsigned int>("threshold");
    RawCollection_  = iConfig.getParameter<edm::InputTag>("rawData");
    // For a list of FEDs by subdetector, see DataFormats/FEDRawData/src/FEDNumbering.cc
    fedStart_       = iConfig.getParameter<unsigned int>("firstFED"); 
    fedStop_        = iConfig.getParameter<unsigned int>("lastFED");
    requireAllFEDs_ = iConfig.getParameter<bool>("requireAllFEDs");
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

    // get the RAW data collction
    edm::Handle<FEDRawDataCollection> theRaw;
    iEvent.getByLabel(RawCollection_, theRaw);
    if (not theRaw.isValid()) {
      edm::LogWarning("HLTFEDSizeFilter") << RawCollection_ << " not available";
      return false;
    }

    bool result = false;

    if (not requireAllFEDs_) {
      // require the at least *one* FED in the given range has size above (or equal to) the threshold
      result = false;
      for (unsigned int i = fedStart_; i <= fedStop_; i++)
        if (theRaw->FEDData(i).size() >= threshold_) {
          result = true;
          break;
        }
    } else {
      // require that *all* FEDs in the given range have size above (or equal to) the threshold
      result = true;
      for (unsigned int i = fedStart_; i <= fedStop_; i++)
        if (theRaw->FEDData(i).size() < threshold_) {
          result = false;
          break;
        }
    }

    return result;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTFEDSizeFilter);
