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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

//
// class declaration
//

class HLTFEDSizeFilter : public HLTFilter {
public:
    explicit HLTFEDSizeFilter(const edm::ParameterSet&);
    ~HLTFEDSizeFilter();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
    virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

    // ----------member data ---------------------------
    edm::EDGetTokenT<FEDRawDataCollection> RawCollectionToken_;
    edm::InputTag RawCollection_;
    unsigned int  threshold_;
    unsigned int  fedStart_, fedStop_ ;
    bool          requireAllFEDs_;

};

//
// constructors and destructor
//
HLTFEDSizeFilter::HLTFEDSizeFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
    threshold_      = iConfig.getParameter<unsigned int>("threshold");
    RawCollection_  = iConfig.getParameter<edm::InputTag>("rawData");
    // For a list of FEDs by subdetector, see DataFormats/FEDRawData/src/FEDNumbering.cc
    fedStart_       = iConfig.getParameter<unsigned int>("firstFED");
    fedStop_        = iConfig.getParameter<unsigned int>("lastFED");
    requireAllFEDs_ = iConfig.getParameter<bool>("requireAllFEDs");

    RawCollectionToken_ = consumes<FEDRawDataCollection>(RawCollection_);
}


HLTFEDSizeFilter::~HLTFEDSizeFilter()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


void
HLTFEDSizeFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("rawData",edm::InputTag("source","",""));
  desc.add<unsigned int>("threshold",0)->
    setComment(" # 0 is pass-through, 1 means *FED ispresent*, higher values are just FED size");
  desc.add<unsigned int>("firstFED",0)->
    setComment(" # first FED, inclusive");
  desc.add<unsigned int>("lastFED",39)->
    setComment(" # last FED, inclusive");
  desc.add<bool>("requireAllFEDs",false)->
    setComment(" # if True, *all* FEDs must be above threshold; if False, only *one* is required");
  descriptions.add("hltFEDSizeFilter",desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTFEDSizeFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {

    // get the RAW data collction
    edm::Handle<FEDRawDataCollection> h_raw;
    iEvent.getByToken(RawCollectionToken_, h_raw);
    // do NOT handle the case where the collection is not available - let the framework handle the exception
    const FEDRawDataCollection theRaw = * h_raw;

    bool result = false;

    if (not requireAllFEDs_) {
      // require that *at least one* FED in the given range has size above or equal to the threshold
      result = false;
      for (unsigned int i = fedStart_; i <= fedStop_; i++)
        if (theRaw.FEDData(i).size() >= threshold_) {
          result = true;
          break;
        }
    } else {
      // require that *all* FEDs in the given range have size above or equal to the threshold
      result = true;
      for (unsigned int i = fedStart_; i <= fedStop_; i++)
        if (theRaw.FEDData(i).size() < threshold_) {
          result = false;
          break;
        }
    }

    return result;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTFEDSizeFilter);
