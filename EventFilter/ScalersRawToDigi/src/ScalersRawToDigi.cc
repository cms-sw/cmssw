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

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

// FEDRawData 
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

// Scalers classes
#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"

class ScalersRawToDigi : public edm::EDProducer 
{
  public:
    explicit ScalersRawToDigi(const edm::ParameterSet&);
    ~ScalersRawToDigi();

    virtual void produce(edm::Event&, const edm::EventSetup&);

  private:
    edm::InputTag inputTag_;
};

// Constructor
ScalersRawToDigi::ScalersRawToDigi(const edm::ParameterSet& iConfig):
  inputTag_((char const *)"source")
{
  produces<L1AcceptBunchCrossingCollection>();
  produces<L1TriggerScalersCollection>();
  produces<Level1TriggerScalersCollection>();
  produces<LumiScalersCollection>();
  if ( iConfig.exists("scalersInputTag") )
  {
    inputTag_ = iConfig.getParameter<edm::InputTag>("scalersInputTag");
  }
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
  iEvent.getByLabel(inputTag_, rawdata);

  std::auto_ptr<LumiScalersCollection> pLumi(new LumiScalersCollection());

  std::auto_ptr<L1TriggerScalersCollection> 
    pOldTrigger(new L1TriggerScalersCollection());

  std::auto_ptr<Level1TriggerScalersCollection> 
    pTrigger(new Level1TriggerScalersCollection());

  std::auto_ptr<L1AcceptBunchCrossingCollection> 
    pBunch(new L1AcceptBunchCrossingCollection());

  /// Take a reference to this FED's data
  const FEDRawData & fedData = rawdata->FEDData(ScalersRaw::SCALERS_FED_ID);
  unsigned short int length =  fedData.size();
  if ( length > 0 ) 
  {
    const ScalersEventRecordRaw_v3 * raw 
	     = (struct ScalersEventRecordRaw_v3 *)fedData.data();
    if ( ( raw->version == 1 ) || ( raw->version == 2 ) )
    {
      L1TriggerScalers oldTriggerScalers(fedData.data());
       pOldTrigger->push_back(oldTriggerScalers);
    }
    else if ( raw->version >= 3 )
    {
      Level1TriggerScalers triggerScalers(fedData.data());
      pTrigger->push_back(triggerScalers);
    }

    LumiScalers      lumiScalers(fedData.data());
    pLumi->push_back(lumiScalers);

    int nWords = length / 8;
    int nBytesExtra = length - sizeof(struct ScalersEventRecordRaw_v1);
    if (( nBytesExtra >= 8 ) && (( nBytesExtra % 8 ) == 0 ))
    {
      unsigned long long * data = 
	(unsigned long long *)fedData.data();

      int nWordsExtra = nBytesExtra / 8;
      for ( int i=0; i<nWordsExtra; i++)
      {
	int index = nWords - 5 + i;
	L1AcceptBunchCrossing bc(i,data[index]);
	pBunch->push_back(bc);
      }
    }
  }
  iEvent.put(pOldTrigger); 
  iEvent.put(pTrigger); 
  iEvent.put(pLumi); 
  iEvent.put(pBunch);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(ScalersRawToDigi);
