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
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/CPUTimer.h"

class SiPixelFedCablingTree;
class SiPixelFedCabling;
class SiPixelQuality;
class TH1D;
class PixelUnpackingRegions;

class SiPixelRawToDigi : public edm::stream::EDProducer<> {
public:

  /// ctor
  explicit SiPixelRawToDigi( const edm::ParameterSet& );

  /// dtor
  virtual ~SiPixelRawToDigi();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// get data, convert to digis attach againe to Event
  virtual void produce( edm::Event&, const edm::EventSetup& ) override;

private:

  edm::ParameterSet config_;
  std::unique_ptr<SiPixelFedCablingTree> cabling_;
  const SiPixelQuality* badPixelInfo_;
  PixelUnpackingRegions* regions_;
  edm::EDGetTokenT<FEDRawDataCollection> tFEDRawDataCollection; 

  TH1D *hCPU, *hDigi;
  std::unique_ptr<edm::CPUTimer> theTimer;
  bool includeErrors;
  bool useQuality;
  bool debug;
  std::vector<int> tkerrorlist;
  std::vector<int> usererrorlist;
  std::vector<unsigned int> fedIds;
  edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher;
  edm::ESWatcher<SiPixelQualityRcd> qualityWatcher;
  edm::InputTag label;
  int ndigis;
  int nwords;
  bool usePilotBlade;
  bool usePhase1;
  std::string CablingMapLabel;
};
#endif
