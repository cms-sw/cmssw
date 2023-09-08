/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*   Nicola Minafra
*   Laurent Forthomme
*
****************************************************************************/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemVFATStatus.h"
#include "DataFormats/CTPPSDigi/interface/TotemFEDInfo.h"

#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "CondFormats/DataRecord/interface/TotemAnalysisMaskRcd.h"
#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"
#include "CondFormats/PPSObjects/interface/TotemAnalysisMask.h"

#include "EventFilter/CTPPSRawToDigi/interface/CTPPSRawToDigiErrorSummary.h"
#include "EventFilter/CTPPSRawToDigi/interface/SimpleVFATFrameCollection.h"
#include "EventFilter/CTPPSRawToDigi/interface/RawDataUnpacker.h"
#include "EventFilter/CTPPSRawToDigi/interface/RawToDigiConverter.h"

#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"
#include "DataFormats/TotemReco/interface/TotemT2Digi.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <string>

class TotemVFATRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit TotemVFATRawToDigi(const edm::ParameterSet &);
  ~TotemVFATRawToDigi() override;

  void produce(edm::Event &, const edm::EventSetup &) override;
  void endStream() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  std::string subSystemName;

  enum { ssUndefined, ssTrackingStrip, ssTimingDiamond, ssTotemTiming, ssTotemT2 } subSystem;

  std::vector<unsigned int> fedIds;

  edm::EDGetTokenT<FEDRawDataCollection> fedDataToken;
  edm::ESGetToken<TotemDAQMapping, TotemReadoutRcd> totemMappingToken;
  edm::ESGetToken<TotemAnalysisMask, TotemAnalysisMaskRcd> analysisMaskToken;

  pps::RawDataUnpacker rawDataUnpacker;
  RawToDigiConverter rawToDigiConverter;
  CTPPSRawToDigiErrorSummary errSummary;

  template <typename DigiType>
  void run(edm::Event &, const edm::EventSetup &);
};

using namespace edm;
using namespace std;

TotemVFATRawToDigi::TotemVFATRawToDigi(const edm::ParameterSet &conf)
    : subSystemName(conf.getParameter<string>("subSystem")),
      subSystem(ssUndefined),
      fedIds(conf.getParameter<vector<unsigned int>>("fedIds")),
      rawDataUnpacker(conf.getParameterSet("RawUnpacking")),
      rawToDigiConverter(conf.getParameterSet("RawToDigi")),
      errSummary("TotemVFATRawToDigi", "[TotemVFATRawToDigi]", false) {
  fedDataToken = consumes<FEDRawDataCollection>(conf.getParameter<edm::InputTag>("rawDataTag"));

  // validate chosen subSystem
  if (subSystemName == "TrackingStrip")
    subSystem = ssTrackingStrip;
  else if (subSystemName == "TimingDiamond")
    subSystem = ssTimingDiamond;
  else if (subSystemName == "TotemTiming")
    subSystem = ssTotemTiming;
  else if (subSystemName == "TotemT2")
    subSystem = ssTotemT2;

  if (subSystem == ssUndefined)
    throw cms::Exception("TotemVFATRawToDigi::TotemVFATRawToDigi")
        << "Unknown sub-system string " << subSystemName << "." << endl;

  // FED (OptoRx) headers and footers
  produces<vector<TotemFEDInfo>>(subSystemName);

  // declare products
  if (subSystem == ssTrackingStrip)
    produces<DetSetVector<TotemRPDigi>>(subSystemName);

  else if (subSystem == ssTimingDiamond)
    produces<DetSetVector<CTPPSDiamondDigi>>(subSystemName);

  else if (subSystem == ssTotemTiming)
    produces<DetSetVector<TotemTimingDigi>>(subSystemName);

  else if (subSystem == ssTotemT2)
    produces<edmNew::DetSetVector<TotemT2Digi>>(subSystemName);

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
      for (int id = FEDNumbering::MINTotemRPTimingVerticalFEDID; id <= FEDNumbering::MAXTotemRPTimingVerticalFEDID;
           ++id)
        fedIds.push_back(id);
    }

    else if (subSystem == ssTotemT2) {
      for (int id = FEDNumbering::MINTotemT2FEDID; id <= FEDNumbering::MAXTotemT2FEDID; ++id)
        fedIds.push_back(id);
    }
  }
  LogDebug("TotemVFATRawToDigi").log([this](auto &log) {
    log << "List of FEDs handled by this instance: ";
    string sep;
    for (const auto &fedId : fedIds)
      log << sep << fedId, sep = ", ";
  });

  // conversion status
  produces<DetSetVector<TotemVFATStatus>>(subSystemName);

  totemMappingToken = esConsumes<TotemDAQMapping, TotemReadoutRcd>(ESInputTag("", subSystemName));
  analysisMaskToken = esConsumes<TotemAnalysisMask, TotemAnalysisMaskRcd>(ESInputTag("", subSystemName));
}

TotemVFATRawToDigi::~TotemVFATRawToDigi() {}

void TotemVFATRawToDigi::produce(edm::Event &event, const edm::EventSetup &es) {
  if (subSystem == ssTrackingStrip)
    run<DetSetVector<TotemRPDigi>>(event, es);

  else if (subSystem == ssTimingDiamond)
    run<DetSetVector<CTPPSDiamondDigi>>(event, es);

  else if (subSystem == ssTotemTiming)
    run<DetSetVector<TotemTimingDigi>>(event, es);

  else if (subSystem == ssTotemT2)
    run<edmNew::DetSetVector<TotemT2Digi>>(event, es);
}

template <typename DigiType>
void TotemVFATRawToDigi::run(edm::Event &event, const edm::EventSetup &es) {
  // get DAQ mapping
  ESHandle<TotemDAQMapping> mapping = es.getHandle(totemMappingToken);
  if (!mapping.isValid() || mapping.failedToGet()) {
    throw cms::Exception("TotemVFATRawToDigi::TotemVFATRawToDigi")
        << "No DAQMapping found for " << subSystemName << "." << endl;
  }

  // get analysis mask to mask channels
  TotemAnalysisMask analysisMask;
  ESHandle<TotemAnalysisMask> analysisMaskHandle = es.getHandle(analysisMaskToken);
  if (analysisMaskHandle.isValid() && !analysisMaskHandle.failedToGet()) {
    analysisMask = *analysisMaskHandle;
  } else {
    errSummary.add(fmt::format("No AnalysisMask found for {0}", subSystemName), "");
    analysisMask = TotemAnalysisMask();
  }

  // raw data handle
  edm::Handle<FEDRawDataCollection> rawData;
  event.getByToken(fedDataToken, rawData);

  // book output products
  vector<TotemFEDInfo> fedInfo;
  DigiType digi;
  DetSetVector<TotemVFATStatus> conversionStatus;

  // raw-data unpacking
  SimpleVFATFrameCollection vfatCollection;
  for (const auto &fedId : fedIds) {
    const FEDRawData &data = rawData->FEDData(fedId);
    if (data.size() > 0)
      rawDataUnpacker.run(fedId, data, fedInfo, vfatCollection);
  }

  // raw-to-digi conversion
  rawToDigiConverter.run(vfatCollection, *mapping, analysisMask, digi, conversionStatus);

  // commit products to event
  event.put(make_unique<vector<TotemFEDInfo>>(fedInfo), subSystemName);
  event.put(make_unique<DigiType>(digi), subSystemName);
  event.put(make_unique<DetSetVector<TotemVFATStatus>>(conversionStatus), subSystemName);
}

void TotemVFATRawToDigi::endStream() {
  rawToDigiConverter.printSummaries();
  errSummary.printSummary();
}

void TotemVFATRawToDigi::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // totemVFATRawToDigi
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("rawDataTag", edm::InputTag(""));
  desc.add<std::string>("subSystem", "")->setComment("options: RP");
  desc.add<std::vector<unsigned int>>("fedIds", {})
      ->setComment(
          "IMPORTANT: leave empty to load the default configuration from "
          "DataFormats/FEDRawData/interface/FEDNumbering.h");
  {
    edm::ParameterSetDescription psd0;
    psd0.addUntracked<unsigned int>("verbosity", 0);
    desc.add<edm::ParameterSetDescription>("RawUnpacking", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.addUntracked<unsigned int>("verbosity", 0)
        ->setComment(
            "0-3: 1=one line/event with some corrupted VFAT frame, 2=list all corrupt VFAT frames/event, 3=all "
            "problems with every corrupt frame");
    psd0.add<unsigned int>("testFootprint", 2)->setComment("0=no test, 1=warn only, 2=warn and skip");
    psd0.add<unsigned int>("testCRC", 2);
    psd0.add<unsigned int>("testID", 2)->setComment("compare the ID from data and mapping");
    psd0.add<unsigned int>("testECMostFrequent", 2)
        ->setComment("compare frame EC with the most frequent value in the event");
    psd0.add<unsigned int>("testBCMostFrequent", 2);
    psd0.addUntracked<unsigned int>("EC_min", 10)
        ->setComment("minimal number of frames to search for the most frequent counter value");
    psd0.addUntracked<unsigned int>("BC_min", 10);
    psd0.addUntracked<double>("EC_fraction", 0.6)
        ->setComment(
            "the most frequent counter value is accepted provided its relative occupancy is higher than this fraction");
    psd0.addUntracked<double>("BC_fraction", 0.6);
    psd0.add<bool>("useOlderT2TestFile", false)
        ->setComment("treat hwID field as two separate 8-bit fields instead of one 16-bit");
    psd0.addUntracked<bool>("printErrorSummary", false)->setComment("per-VFAT error summary at the end of the job");
    psd0.addUntracked<bool>("printUnknownFrameSummary", false)
        ->setComment("summary of frames found in data, but not in the mapping");
    desc.add<edm::ParameterSetDescription>("RawToDigi", psd0);
  }
  descriptions.add("totemVFATRawToDigi", desc);
  // or use the following to generate the label from the module's C++ type
  //descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(TotemVFATRawToDigi);
