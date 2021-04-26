#ifndef SiPixelRawToDigi_H
#define SiPixelRawToDigi_H

/** \class SiPixelRawToDigi_H
 *  Plug-in module that performs Raw data to digi conversion 
 *  for pixel subdetector
 */

#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

class SiPixelFedCablingTree;
class SiPixelFedCabling;
class SiPixelQuality;
class TH1D;
class PixelUnpackingRegions;

class SiPixelRawToDigi : public edm::stream::EDProducer<> {
public:
  /// ctor
  explicit SiPixelRawToDigi(const edm::ParameterSet&);

  /// dtor
  ~SiPixelRawToDigi() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// get data, convert to digis attach againe to Event
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::ParameterSet config_;
  std::unique_ptr<SiPixelFedCablingTree> cabling_;
  const SiPixelQuality* badPixelInfo_;
  PixelUnpackingRegions* regions_;
  const std::vector<int> tkerrorlist_;
  const std::vector<int> usererrorlist_;
  std::vector<unsigned int> fedIds_;
  edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher_;
  edm::ESWatcher<SiPixelQualityRcd> qualityWatcher_;
  const edm::EDGetTokenT<FEDRawDataCollection> fedRawDataCollectionToken_;
  edm::ESGetToken<SiPixelQuality, SiPixelQualityRcd> siPixelQualityToken_;
  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> cablingMapToken_;
  const bool includeErrors_;
  const bool useQuality_;
  const bool usePilotBlade_;
  const bool usePhase1_;
};
#endif
