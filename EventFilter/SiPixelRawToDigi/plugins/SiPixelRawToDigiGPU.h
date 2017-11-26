#ifndef SiPixelRawToDigiGPU_H
#define SiPixelRawToDigiGPU_H

/** \class SiPixelRawToDigiGPU_H
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

class SiPixelRawToDigiGPU : public edm::stream::EDProducer<> {
public:
 
  /// ctor
  explicit SiPixelRawToDigiGPU( const edm::ParameterSet& );

  /// dtor
  ~SiPixelRawToDigiGPU() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// get data, convert to digis attach againe to Event

  void produce( edm::Event&, const edm::EventSetup& ) override;

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
  std::string cablingMapLabel;
  typedef cms_uint32_t Word32;
  typedef cms_uint64_t Word64;
  
  bool convertADCtoElectrons;
  unsigned int *word;        // to hold input for rawtodigi
  unsigned int *fedIndex;    // to hold fed index inside word[] array for rawtodigi on GPU
  unsigned int *eventIndex;  // to store staring index of each event in word[] array

  // to store the output
  // uint *word_h, *fedIndex_h, *eventIndex_h;       // host copy of input data
  uint *xx_h, *yy_h, *adc_h, *rawIdArr_h;  // host copy of output
  // store the start and end index for each module (total 1856 modules-phase 1)
  int *mIndexStart_h, *mIndexEnd_h; 
  
};
#endif
