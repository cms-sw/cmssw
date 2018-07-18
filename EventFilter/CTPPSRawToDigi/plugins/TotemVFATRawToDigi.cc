/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/CTPPSDigi/interface/TotemFEDInfo.h"

#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemDAQMapping.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemAnalysisMask.h"

#include "EventFilter/CTPPSRawToDigi/interface/SimpleVFATFrameCollection.h"
#include "EventFilter/CTPPSRawToDigi/interface/RawDataUnpacker.h"
#include "EventFilter/CTPPSRawToDigi/interface/RawToDigiConverter.h"

#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class TotemVFATRawToDigi : public edm::stream::EDProducer<>
{
  public:
    explicit TotemVFATRawToDigi(const edm::ParameterSet&);
    ~TotemVFATRawToDigi() override;

    void produce(edm::Event&, const edm::EventSetup&) override;
    void endStream() override;

  private:
    std::string subSystemName;

    enum { ssUndefined, ssTrackingStrip, ssTimingDiamond, ssTotemTiming } subSystem;

    std::vector<unsigned int> fedIds;

    edm::EDGetTokenT<FEDRawDataCollection> fedDataToken;

    ctpps::RawDataUnpacker rawDataUnpacker;
    RawToDigiConverter rawToDigiConverter;

    template <typename DigiType>
    void run(edm::Event&, const edm::EventSetup&);
};

//----------------------------------------------------------------------------------------------------

using namespace edm;
using namespace std;

//----------------------------------------------------------------------------------------------------

TotemVFATRawToDigi::TotemVFATRawToDigi(const edm::ParameterSet &conf):
  subSystemName(conf.getParameter<string>("subSystem")),
  subSystem(ssUndefined),
  fedIds(conf.getParameter< vector<unsigned int> >("fedIds")),
  rawDataUnpacker(conf.getParameterSet("RawUnpacking")),
  rawToDigiConverter(conf.getParameterSet("RawToDigi"))
{
  fedDataToken = consumes<FEDRawDataCollection>(conf.getParameter<edm::InputTag>("rawDataTag"));

  // validate chosen subSystem
  if (subSystemName == "TrackingStrip")
    subSystem = ssTrackingStrip;
  else if (subSystemName == "TimingDiamond")
    subSystem = ssTimingDiamond;
  else if (subSystemName == "TotemTiming")
    subSystem = ssTotemTiming;

  if (subSystem == ssUndefined)
    throw cms::Exception("TotemVFATRawToDigi::TotemVFATRawToDigi") << "Unknown sub-system string " << subSystemName << "." << endl;

  // FED (OptoRx) headers and footers
  produces< vector<TotemFEDInfo> >(subSystemName);

  // declare products
  if (subSystem == ssTrackingStrip)
    produces< DetSetVector<TotemRPDigi> >(subSystemName);

  else if (subSystem == ssTimingDiamond)
    produces< DetSetVector<CTPPSDiamondDigi> >(subSystemName);

  else if (subSystem == ssTotemTiming)
    produces< DetSetVector<TotemTimingDigi> >(subSystemName);

  // set default IDs
  if (fedIds.empty()) {
    if (subSystem == ssTrackingStrip) {
      for (int id = FEDNumbering::MINTotemRPHorizontalFEDID; id <= FEDNumbering::MAXTotemRPHorizontalFEDID; ++id)
        fedIds.push_back(id);

      for (int id = FEDNumbering::MINTotemRPVerticalFEDID; id <= FEDNumbering::MAXTotemRPVerticalFEDID; ++id)
        fedIds.push_back(id);
    }

    else if (subSystem == ssTimingDiamond) {
      for (int id = FEDNumbering::MINCTPPSDiamondFEDID; id <= FEDNumbering::MAXCTPPSDiamondFEDID; ++id)
        fedIds.push_back(id);
    }

    else if (subSystem == ssTotemTiming) {
      for (int id = FEDNumbering::MINTotemRPTimingVerticalFEDID; id <= FEDNumbering::MAXTotemRPTimingVerticalFEDID; ++id)
        fedIds.push_back(id);
    }
  }

  // conversion status
  produces< DetSetVector<TotemVFATStatus> >(subSystemName);
}

//----------------------------------------------------------------------------------------------------

TotemVFATRawToDigi::~TotemVFATRawToDigi()
{
}

//----------------------------------------------------------------------------------------------------

void TotemVFATRawToDigi::produce(edm::Event& event, const edm::EventSetup &es)
{
  if (subSystem == ssTrackingStrip)
    run< DetSetVector<TotemRPDigi> >(event, es);

  else if (subSystem == ssTimingDiamond)
    run< DetSetVector<CTPPSDiamondDigi> >(event, es);

  else if (subSystem == ssTotemTiming)
    run< DetSetVector<TotemTimingDigi> >(event, es);
}

//----------------------------------------------------------------------------------------------------

template <typename DigiType>
void TotemVFATRawToDigi::run(edm::Event& event, const edm::EventSetup &es)
{
  // get DAQ mapping
  ESHandle<TotemDAQMapping> mapping;
  es.get<TotemReadoutRcd>().get(subSystemName, mapping);

  // get analysis mask to mask channels
  ESHandle<TotemAnalysisMask> analysisMask;
  es.get<TotemReadoutRcd>().get(subSystemName, analysisMask);

  // raw data handle
  edm::Handle<FEDRawDataCollection> rawData;
  event.getByToken(fedDataToken, rawData);

  // book output products
  vector<TotemFEDInfo> fedInfo;
  DigiType digi;
  DetSetVector<TotemVFATStatus> conversionStatus;

  // raw-data unpacking
  SimpleVFATFrameCollection vfatCollection;
  for (const auto &fedId : fedIds)
  {
    const FEDRawData &data = rawData->FEDData(fedId);
    if (data.size() > 0)
      rawDataUnpacker.run(fedId, data, fedInfo, vfatCollection);
  }

  // raw-to-digi conversion
  rawToDigiConverter.run(vfatCollection, *mapping, *analysisMask, digi, conversionStatus);

  // commit products to event
  event.put(make_unique<vector<TotemFEDInfo>>(fedInfo), subSystemName);
  event.put(make_unique<DigiType>(digi), subSystemName);
  event.put(make_unique<DetSetVector<TotemVFATStatus>>(conversionStatus), subSystemName);
}

//----------------------------------------------------------------------------------------------------

void TotemVFATRawToDigi::endStream()
{
  rawToDigiConverter.printSummaries();
}

DEFINE_FWK_MODULE(TotemVFATRawToDigi);
