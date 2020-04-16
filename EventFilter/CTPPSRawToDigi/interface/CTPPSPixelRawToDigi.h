#ifndef CTPPS_CTPPSPixelRawToDigi_CTPPSPixelRawToDigi_h
#define CTPPS_CTPPSPixelRawToDigi_CTPPSPixelRawToDigi_h

/** \class CTPPSPixelRawToDigi_H
 *  Plug-in module that performs Raw data to digi conversion 
 *  for CTPPS pixel detector
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelDAQMapping.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class CTPPSPixelRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit CTPPSPixelRawToDigi(const edm::ParameterSet&);

  ~CTPPSPixelRawToDigi() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// get data, convert to digis attach againe to Event
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::ParameterSet config_;

  edm::EDGetTokenT<FEDRawDataCollection> FEDRawDataCollection_;

  std::set<unsigned int> fedIds_;

  edm::InputTag label_;

  std::string mappingLabel_;

  bool includeErrors_;
};
#endif
