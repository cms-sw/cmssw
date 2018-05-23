#ifndef SiPixelRawToDigiGPU_H
#define SiPixelRawToDigiGPU_H

/** \class SiPixelRawToDigiGPU_H
 *  Plug-in module that performs Raw data to digi conversion
 *  for pixel subdetector
 */

#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTService.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/CPUTimer.h"
#include "SiPixelRawToDigiGPUKernel.h"

class PixelUnpackingRegions;
class SiPixelFedCablingTree;
class SiPixelFedCabling;
class SiPixelQuality;
class TH1D;
class SiPixelGainForHLTonGPU;
struct SiPixelGainForHLTonGPU_DecodingStructure;

class SiPixelRawToDigiGPU : public edm::stream::EDProducer<>
{
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
  const TrackerTopology* tTopo_;          // needed to get correct layer number

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

  bool convertADCtoElectrons;
  unsigned int *word;        // to hold input for rawtodigi
  unsigned char *fedId_h;    // to hold fed index for each word

  // to store the output
  uint32_t *pdigi_h, *rawIdArr_h;                   // host copy of output
  uint16_t * adc_h; int32_t * clus_h; // host copy of calib&clus output
  pixelgpudetails::error_obj *data_h = nullptr;
  GPU::SimpleVector<pixelgpudetails::error_obj> *error_h = nullptr;
  GPU::SimpleVector<pixelgpudetails::error_obj> *error_h_tmp = nullptr;

  // configuration and memory buffers alocated on the GPU
  pixelgpudetails::context context_;
  SiPixelFedCablingMapGPU * cablingMapGPUHost_;
  SiPixelFedCablingMapGPU * cablingMapGPUDevice_;

  //  gain calib
  SiPixelGainCalibrationForHLTService  theSiPixelGainCalibration_;
  SiPixelGainForHLTonGPU  * gainForHLTonGPU_ = nullptr;
  SiPixelGainForHLTonGPU_DecodingStructure * gainDataOnGPU_ = nullptr;

};

#endif
