// -*- C++ -*-
//
// Package:    ScalersRawToDigi
// Class:      ScalersRawToDigi
// 
/**\class ScalersRawToDigi ScalersRawToDigi.cc EventFilter/ScalersRawToDigi/src/ScalersRawToDigi.cc

 Description: Unpack FED data to Trigger and Lumi Scalers bank. 
 These Scalers are in FED id xxx

 Implementation:
     No comments
*/
//
// Original Author:  William Badgett
//         Created:  Wed Nov 14 07:47:59 CDT 2006
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//FEDRawData 
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

// Scalers classes
#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"

//
// class declaration
//

class ScalersRawToDigi : public edm::EDProducer 
{
  public:
    explicit ScalersRawToDigi(const edm::ParameterSet&);
    ~ScalersRawToDigi();

    virtual void produce(edm::Event&, const edm::EventSetup&);
  private:
    // ----------member data ---------------------------
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
ScalersRawToDigi::ScalersRawToDigi(const edm::ParameterSet& iConfig)
{
  //register your products
  produces<L1TriggerScalers>();
  produces<LumiScalers>();
}


ScalersRawToDigi::~ScalersRawToDigi()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void ScalersRawToDigi::produce(edm::Event& iEvent, 
			       const edm::EventSetup& iSetup)
{
  using namespace edm;
  const int ScalersFedID = 999;

  // Get a handle to the FED data collection
  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByLabel("source" , rawdata);

  std::auto_ptr<LumiScalersCollection> pLumi(new LumiScalersCollection(1));
  std::auto_ptr<L1TriggerScalersCollection> 
    pTrigger(new L1TriggerScalersCollection(1));

  /// Take a reference to this FED's data
  const FEDRawData & fedData = rawdata->FEDData(ScalersFedID);
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

//define this as a plug-in
DEFINE_FWK_MODULE(ScalersRawToDigi);
