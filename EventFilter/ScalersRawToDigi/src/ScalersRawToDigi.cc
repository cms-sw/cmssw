// -*- C++ -*-
//
// Package:    EventFilter/ScalersRawToDigi
// Class:      ScalersRawToDigi
// 
/**\class ScalersRawToDigi ScalersRawToDigi.cc EventFilter/ScalersRawToDigi/src/ScalersRawToDigi.cc

 Description: Unpack FED data to Trigger and Lumi Scalers "bank"
 These Scalers are in FED id ScalersRaw::SCALERS_FED_ID

*/
//
// Original Author:  William Badgett
//         Created:  Wed Nov 14 07:47:59 CDT 2006
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// FEDRawData 
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

// Scalers classes
#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"

class ScalersRawToDigi : public edm::EDProducer 
{
  public:
    explicit ScalersRawToDigi(const edm::ParameterSet&);
    ~ScalersRawToDigi();

    virtual void produce(edm::Event&, const edm::EventSetup&);
};

// Constructor
ScalersRawToDigi::ScalersRawToDigi(const edm::ParameterSet& iConfig)
{
  produces<L1TriggerScalers>();
  produces<LumiScalers>();
}

// Destructor
ScalersRawToDigi::~ScalersRawToDigi() {}

// Method called to produce the data 
void ScalersRawToDigi::produce(edm::Event& iEvent, 
			       const edm::EventSetup& iSetup)
{
  using namespace edm;

  // Get a handle to the FED data collection
  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByLabel("source" , rawdata);

  std::auto_ptr<LumiScalersCollection> pLumi(new LumiScalersCollection(1));
  std::auto_ptr<L1TriggerScalersCollection> 
    pTrigger(new L1TriggerScalersCollection(1));

  /// Take a reference to this FED's data
  const FEDRawData & fedData = rawdata->FEDData(ScalersRaw::SCALERS_FED_ID);
  unsigned short int length =  fedData.size();
  if ( length > 0 ) 
  {
    L1TriggerScalers triggerScalers(fedData.data());
    LumiScalers      lumiScalers(fedData.data());
    pLumi->push_back(lumiScalers);
    pTrigger->push_back(triggerScalers);
    iEvent.put(pLumi); 
    iEvent.put(pTrigger); 
  }
}

// Define this as a plug-in
DEFINE_FWK_MODULE(ScalersRawToDigi);
