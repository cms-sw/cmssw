/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/TotemRPDigi/interface/TotemRPDigi.h"
#include "DataFormats/TotemRPL1/interface/TotemRPCCBits.h"
#include "DataFormats/TotemRawData/interface/TotemRawEvent.h"
#include "DataFormats/TotemRawData/interface/TotemRawToDigiStatus.h"

#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "CondFormats/TotemReadoutObjects/interface/TotemDAQMapping.h"
#include "CondFormats/TotemReadoutObjects/interface/TotemAnalysisMask.h"

#include "EventFilter/TotemRawToDigi/interface/SimpleVFATFrameCollection.h"
#include "EventFilter/TotemRawToDigi/interface/RawDataUnpacker.h"
#include "EventFilter/TotemRawToDigi/interface/RawToDigiConverter.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class TotemRawToDigi : public edm::EDProducer 
{
  public:
    explicit TotemRawToDigi(const edm::ParameterSet&);
    ~TotemRawToDigi();

    virtual void produce(edm::Event&, const edm::EventSetup&) override;
    virtual void endJob();

  private:
    edm::EDGetTokenT<FEDRawDataCollection> fedDataToken;

    /// product labels
    std::string rpDataProductLabel;
    std::string rpCCProductLabel;
    std::string conversionStatusLabel;

    RawDataUnpacker rawDataUnpacker;
    RawToDigiConverter rawToDigiConverter;
};

//----------------------------------------------------------------------------------------------------

using namespace edm;
using namespace std;

//----------------------------------------------------------------------------------------------------

TotemRawToDigi::TotemRawToDigi(const edm::ParameterSet &conf):
  rawDataUnpacker(conf.getParameterSet("RawUnpacking")),
  rawToDigiConverter(conf.getParameterSet("RawToDigi"))
{
  fedDataToken = consumes<FEDRawDataCollection>(conf.getParameter<edm::InputTag>("rawDataTag"));

  produces<TotemRawEvent>();

  // RP data
  rpDataProductLabel = conf.getUntrackedParameter<std::string>("rpDataProductLabel", "");
  produces< edm::DetSetVector<TotemRPDigi> > (rpDataProductLabel);

  // RP CC
  rpCCProductLabel = conf.getUntrackedParameter<std::string>("rpCCProductLabel", "");
  produces < std::vector <TotemRPCCBits> > (rpCCProductLabel);

  // status
  conversionStatusLabel = conf.getUntrackedParameter<std::string>("conversionStatusLabel", "");
  produces <TotemRawToDigiStatus>(conversionStatusLabel);
}

//----------------------------------------------------------------------------------------------------

TotemRawToDigi::~TotemRawToDigi()
{
}

//----------------------------------------------------------------------------------------------------

void TotemRawToDigi::produce(edm::Event& event, const edm::EventSetup &es)
{
  // get DAQ mapping
  ESHandle<TotemDAQMapping> mapping;
  es.get<TotemReadoutRcd>().get(mapping);

  // get analysis mask to mask channels
  ESHandle<TotemAnalysisMask> analysisMask;
  es.get<TotemReadoutRcd>().get(analysisMask);

  // raw data handle
  edm::Handle<FEDRawDataCollection> rawData;
  event.getByToken(fedDataToken, rawData);

  // book output products
  auto_ptr<TotemRawEvent> totemRawEvent(new TotemRawEvent);

  auto_ptr< DetSetVector<TotemRPDigi> > rpDataOutput(new edm::DetSetVector<TotemRPDigi>);  
  auto_ptr< vector<TotemRPCCBits> > rpCCOutput(new std::vector<TotemRPCCBits>);
  auto_ptr< TotemRawToDigiStatus > conversionStatus(new TotemRawToDigiStatus());

  // step 1: raw-data unpacking
  SimpleVFATFrameCollection vfatCollection;

  // TODO: replace with real FED Ids
  /* Hints from Michele
      DEVICE      ID     DETECTOR          OLD ID
      Trigger     577    LONEG             0x29c
      RX 1        578    5-6 210m FAR      0x1a1
      RX 2        579    5-6 210m NEAR     0x1a2
      RX 3        580    4-5 210m FAR      0x1a9
      RX 4        581    4-5 210m NEAR     0x1aa  
  */
  /* 
      The FED Ids should be stored in and read from:
        DataFormats/FEDRawData/interface/FEDNumbering.h
  */
  vector<int> fedIds = { 0, 1, 2, 3, 4, 5, 6 }; 

  for (const auto &fedId : fedIds)
  {
    rawDataUnpacker.Run(fedId, rawData->FEDData(fedId), vfatCollection, *totemRawEvent);
  }

  // step 2: raw to digi
  rawToDigiConverter.Run(vfatCollection, *mapping, *analysisMask,
    *rpDataOutput, *rpCCOutput, *conversionStatus);

  // commit products to event
  event.put(totemRawEvent);
  event.put(rpDataOutput, rpDataProductLabel);
  event.put(rpCCOutput, rpCCProductLabel);
  event.put(conversionStatus, conversionStatusLabel);
}

//----------------------------------------------------------------------------------------------------

void TotemRawToDigi::endJob()
{
  rawToDigiConverter.PrintSummaries();
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(TotemRawToDigi);
