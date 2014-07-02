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

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"

// FEDRawData 
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

// Scalers classes
#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DataFormats/Scalers/interface/L1TriggerScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/Scalers/interface/Level1TriggerRates.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"

class ScalersRawToDigi : public edm::stream::EDProducer<> 
{
  public:
    explicit ScalersRawToDigi(const edm::ParameterSet&);
    ~ScalersRawToDigi();

    virtual void produce(edm::Event&, const edm::EventSetup&) override;

  private:
    edm::InputTag inputTag_;
    edm::EDGetTokenT<FEDRawDataCollection> fedToken_;

};

// Constructor
ScalersRawToDigi::ScalersRawToDigi(const edm::ParameterSet& iConfig):
  inputTag_((char const *)"rawDataCollector")
{
  produces<L1AcceptBunchCrossingCollection>();
  produces<L1TriggerScalersCollection>();
  produces<Level1TriggerScalersCollection>();
  produces<LumiScalersCollection>();
  produces<BeamSpotOnlineCollection>();
  produces<DcsStatusCollection>();
  if ( iConfig.exists("scalersInputTag") )
  {
    inputTag_ = iConfig.getParameter<edm::InputTag>("scalersInputTag");
  }
  fedToken_=consumes<FEDRawDataCollection>(inputTag_);

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
  iEvent.getByToken(fedToken_, rawdata);

  std::auto_ptr<LumiScalersCollection> pLumi(new LumiScalersCollection());

  std::auto_ptr<L1TriggerScalersCollection> 
    pOldTrigger(new L1TriggerScalersCollection());

  std::auto_ptr<Level1TriggerScalersCollection> 
    pTrigger(new Level1TriggerScalersCollection());

  std::auto_ptr<L1AcceptBunchCrossingCollection> 
    pBunch(new L1AcceptBunchCrossingCollection());

  std::auto_ptr<BeamSpotOnlineCollection> pBeamSpotOnline(new BeamSpotOnlineCollection());
  std::auto_ptr<DcsStatusCollection> pDcsStatus(new DcsStatusCollection());

  /// Take a reference to this FED's data
  const FEDRawData & fedData = rawdata->FEDData(ScalersRaw::SCALERS_FED_ID);
  unsigned short int length =  fedData.size();
  if ( length > 0 ) 
  {
    int nWords = length / 8;
    int nBytesExtra = 0;

    const ScalersEventRecordRaw_v6 * raw 
	     = (struct ScalersEventRecordRaw_v6 *)fedData.data();
    if ( ( raw->version == 1 ) || ( raw->version == 2 ) )
    {
      L1TriggerScalers oldTriggerScalers(fedData.data());
      pOldTrigger->push_back(oldTriggerScalers);
      nBytesExtra = length - sizeof(struct ScalersEventRecordRaw_v1);
    }
    else if ( raw->version >= 3 )
    {
      Level1TriggerScalers triggerScalers(fedData.data());
      pTrigger->push_back(triggerScalers);
      if ( raw->version >= 6 )
      {
	nBytesExtra = ScalersRaw::N_BX_v6 * sizeof(unsigned long long);
      }
      else
      {
	nBytesExtra = ScalersRaw::N_BX_v2 * sizeof(unsigned long long);
      }
    }

    LumiScalers      lumiScalers(fedData.data());
    pLumi->push_back(lumiScalers);

    if (( nBytesExtra >= 8 ) && (( nBytesExtra % 8 ) == 0 ))
    {
      unsigned long long * data = 
	(unsigned long long *)fedData.data();

      int nWordsExtra = nBytesExtra / 8;
      for ( int i=0; i<nWordsExtra; i++)
      {
	int index = nWords - (nWordsExtra + 1) + i;
	L1AcceptBunchCrossing bc(i,data[index]);
	pBunch->push_back(bc);
      }
    }

    if ( raw->version >= 4 )
    {
      BeamSpotOnline beamSpotOnline(fedData.data());
      pBeamSpotOnline->push_back(beamSpotOnline);

      DcsStatus dcsStatus(fedData.data());
      pDcsStatus->push_back(dcsStatus);
    }
  }
  iEvent.put(pOldTrigger); 
  iEvent.put(pTrigger); 
  iEvent.put(pLumi); 
  iEvent.put(pBunch);
  iEvent.put(pBeamSpotOnline);
  iEvent.put(pDcsStatus);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(ScalersRawToDigi);
