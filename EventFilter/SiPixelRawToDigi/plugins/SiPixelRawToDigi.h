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
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

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
  // always consumed
  const edm::EDGetTokenT<FEDRawDataCollection> fedRawDataCollectionToken_;
  const edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> cablingMapToken_;
  // consume only if pixel quality is used -> useQuality_
  edm::ESGetToken<SiPixelQuality, SiPixelQualityRcd> siPixelQualityToken_;
  // always produced
  edm::EDPutTokenT<edm::DetSetVector<PixelDigi>> siPixelDigiCollectionToken_;
  // produce only if error collections are included -> includeErrors_
  edm::EDPutTokenT<edm::DetSetVector<SiPixelRawDataError>> errorPutToken_;
  edm::EDPutTokenT<DetIdCollection> tkErrorPutToken_;
  edm::EDPutTokenT<DetIdCollection> userErrorPutToken_;
  edm::EDPutTokenT<edmNew::DetSetVector<PixelFEDChannel>> disabledChannelPutToken_;
  const bool includeErrors_;
  const bool useQuality_;
  const bool usePilotBlade_;
  const bool usePhase1_;
};
#endif
