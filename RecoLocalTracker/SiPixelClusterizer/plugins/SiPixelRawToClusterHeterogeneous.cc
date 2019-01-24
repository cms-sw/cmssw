// C++ includes
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>

// CUDA kincludes
#include <cuda.h>
#include <cuda_runtime.h>

// CMSSW includes
#include "CalibTracker/Records/interface/SiPixelGainCalibrationForHLTGPURcd.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTGPU.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTService.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelUnpackingRegions.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelFedCablingMapGPUWrapper.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "SiPixelRawToClusterGPUKernel.h"
#include "siPixelRawToClusterHeterogeneousProduct.h"
#include "PixelThresholdClusterizer.h"

namespace {
  struct AccretionCluster {
    typedef unsigned short UShort;
    static constexpr UShort MAXSIZE = 256;
    UShort adc[MAXSIZE];
    UShort x[MAXSIZE];
    UShort y[MAXSIZE];
    UShort xmin=16000;
    UShort ymin=16000;
    unsigned int isize=0;
    int charge=0;

    void clear() {
      isize=0;
      charge=0;
      xmin=16000;
      ymin=16000;
    }

    bool add(SiPixelCluster::PixelPos const & p, UShort const iadc) {
      if (isize==MAXSIZE) return false;
      xmin=std::min(xmin,(unsigned short)(p.row()));
      ymin=std::min(ymin,(unsigned short)(p.col()));
      adc[isize]=iadc;
      x[isize]=p.row();
      y[isize++]=p.col();
      charge+=iadc;
      return true;
    }
  };

  constexpr uint32_t dummydetid = 0xffffffff;
}

class SiPixelRawToClusterHeterogeneous: public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices <
                                                                      heterogeneous::GPUCuda,
                                                                      heterogeneous::CPU
                                                                      > > {
public:
  using CPUProduct = siPixelRawToClusterHeterogeneousProduct::CPUProduct;
  using GPUProduct = siPixelRawToClusterHeterogeneousProduct::GPUProduct;
  using Output = siPixelRawToClusterHeterogeneousProduct::HeterogeneousDigiCluster;

  explicit SiPixelRawToClusterHeterogeneous(const edm::ParameterSet& iConfig);
  ~SiPixelRawToClusterHeterogeneous() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // CPU implementation
  void produceCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) override;

  // GPU implementation
  void acquireGPUCuda(const edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, cuda::stream_t<>& cudaStream) override;
  void produceGPUCuda(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, cuda::stream_t<>& cudaStream) override;
  void convertGPUtoCPU(edm::Event& ev, unsigned int nDigis, pixelgpudetails::SiPixelRawToClusterGPUKernel::CPUData) const;

  // Commonalities
  const FEDRawDataCollection *initialize(const edm::Event& ev, const edm::EventSetup& es);

  std::unique_ptr<SiPixelFedCablingTree> cabling_;
  const SiPixelQuality *badPixelInfo_ = nullptr;
  const SiPixelFedCablingMap *cablingMap_ = nullptr;
std::unique_ptr<PixelUnpackingRegions> regions_;
  edm::EDGetTokenT<FEDRawDataCollection> tFEDRawDataCollection;

  bool includeErrors;
  bool useQuality;
  bool debug;
  std::vector<int> tkerrorlist;
  std::vector<int> usererrorlist;
  std::vector<unsigned int> fedIds;

  edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher;
  edm::ESWatcher<SiPixelQualityRcd> qualityWatcher;

  bool usePilotBlade;
  bool usePhase1;
  bool convertADCtoElectrons;
  std::string cablingMapLabel;

  // clusterizer
  PixelThresholdClusterizer clusterizer_;
  const TrackerGeometry *geom_ = nullptr;
  const TrackerTopology *ttopo_ = nullptr;

  //  gain calib
  SiPixelGainCalibrationForHLTService  theSiPixelGainCalibration_;

  // GPU algo
  pixelgpudetails::SiPixelRawToClusterGPUKernel gpuAlgo_;
  PixelDataFormatter::Errors errors_;

  bool enableTransfer_;
  bool enableConversion_;
};

SiPixelRawToClusterHeterogeneous::SiPixelRawToClusterHeterogeneous(const edm::ParameterSet& iConfig):
  HeterogeneousEDProducer(iConfig),
  clusterizer_(iConfig),
  theSiPixelGainCalibration_(iConfig) {
  includeErrors = iConfig.getParameter<bool>("IncludeErrors");
  useQuality = iConfig.getParameter<bool>("UseQualityInfo");
  tkerrorlist = iConfig.getParameter<std::vector<int> > ("ErrorList");
  usererrorlist = iConfig.getParameter<std::vector<int> > ("UserErrorList");
  tFEDRawDataCollection = consumes <FEDRawDataCollection> (iConfig.getParameter<edm::InputTag>("InputLabel"));

  enableConversion_ = iConfig.getParameter<bool>("gpuEnableConversion");
  enableTransfer_ = enableConversion_ || iConfig.getParameter<bool>("gpuEnableTransfer");

  clusterizer_.setSiPixelGainCalibrationService(&theSiPixelGainCalibration_);

  // Products in GPU
  produces<HeterogeneousProduct>();
  // Products in CPU
  if(enableConversion_) {
    produces<edm::DetSetVector<PixelDigi>>();
    if(includeErrors) {
      produces<edm::DetSetVector<SiPixelRawDataError>>();
      produces<DetIdCollection>();
      produces<DetIdCollection>("UserErrorModules");
      produces<SiPixelClusterCollectionNew>();
      produces<edmNew::DetSetVector<PixelFEDChannel>>();
    }
  }

  // regions
  if(!iConfig.getParameter<edm::ParameterSet>("Regions").getParameterNames().empty()) {
    regions_ = std::make_unique<PixelUnpackingRegions>(iConfig, consumesCollector());
  }

  // Control the usage of pilot-blade data, FED=40
  usePilotBlade = iConfig.getParameter<bool> ("UsePilotBlade");
  if(usePilotBlade) edm::LogInfo("SiPixelRawToCluster")  << " Use pilot blade data (FED 40)";

  // Control the usage of phase1
  usePhase1 = iConfig.getParameter<bool> ("UsePhase1");
  if(usePhase1) edm::LogInfo("SiPixelRawToCluster")  << " Using phase1";

  //CablingMap could have a label //Tav
  cablingMapLabel = iConfig.getParameter<std::string> ("CablingMapLabel");

  convertADCtoElectrons = iConfig.getParameter<bool>("ConvertADCtoElectrons");
}

void SiPixelRawToClusterHeterogeneous::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("IncludeErrors",true);
  desc.add<bool>("UseQualityInfo",false);
  {
    std::vector<int> temp1;
    temp1.reserve(1);
    temp1.push_back(29);
    desc.add<std::vector<int> >("ErrorList",temp1)->setComment("## ErrorList: list of error codes used by tracking to invalidate modules");
  }
  {
    std::vector<int> temp1;
    temp1.reserve(1);
    temp1.push_back(40);
    desc.add<std::vector<int> >("UserErrorList",temp1)->setComment("## UserErrorList: list of error codes used by Pixel experts for investigation");
  }
  desc.add<edm::InputTag>("InputLabel",edm::InputTag("rawDataCollector"));
  {
    edm::ParameterSetDescription psd0;
    psd0.addOptional<std::vector<edm::InputTag>>("inputs");
    psd0.addOptional<std::vector<double>>("deltaPhi");
    psd0.addOptional<std::vector<double>>("maxZ");
    psd0.addOptional<edm::InputTag>("beamSpot");
    desc.add<edm::ParameterSetDescription>("Regions",psd0)->setComment("## Empty Regions PSet means complete unpacking");
  }
  desc.add<bool>("UsePilotBlade",false)->setComment("##  Use pilot blades");
  desc.add<bool>("UsePhase1",false)->setComment("##  Use phase1");
  desc.add<std::string>("CablingMapLabel","")->setComment("CablingMap label"); //Tav
  desc.addOptional<bool>("CheckPixelOrder");  // never used, kept for back-compatibility

  desc.add<bool>("ConvertADCtoElectrons", false)->setComment("## do the calibration ADC-> Electron and apply the threshold, requried for clustering");

  // clusterizer
  desc.add<int>("ChannelThreshold", 1000);
  desc.add<int>("SeedThreshold", 1000);
  desc.add<int>("ClusterThreshold", 4000);
  desc.add<int>("ClusterThreshold_L1", 4000);
  desc.add<int>("VCaltoElectronGain", 65);
  desc.add<int>("VCaltoElectronGain_L1", 65);
  desc.add<int>("VCaltoElectronOffset", -414);
  desc.add<int>("VCaltoElectronOffset_L1", -414);
  desc.addUntracked<bool>("MissCalibrate", true);
  desc.add<bool>("SplitClusters", false);
  desc.add<double>("ElectronPerADCGain", 135.);
  // Phase 2 clusterizer
  desc.add<bool>("Phase2Calibration", false);
  desc.add<int>("Phase2ReadoutMode", -1);
  desc.add<double>("Phase2DigiBaseline", 1200.);
  desc.add<int>("Phase2KinkADC", 8);

  desc.add<bool>("gpuEnableTransfer", true);
  desc.add<bool>("gpuEnableConversion", true);

  HeterogeneousEDProducer::fillPSetDescription(desc);

  descriptions.add("siPixelClustersHeterogeneousDefault",desc);
}

const FEDRawDataCollection *SiPixelRawToClusterHeterogeneous::initialize(const edm::Event& ev, const edm::EventSetup& es) {
  debug = edm::MessageDrop::instance()->debugEnabled;

  // setup gain calibration service
  theSiPixelGainCalibration_.setESObjects( es );

  // initialize cabling map or update if necessary
  if (recordWatcher.check( es )) {
    // cabling map, which maps online address (fed->link->ROC->local pixel) to offline (DetId->global pixel)
    edm::ESTransientHandle<SiPixelFedCablingMap> cablingMap;
    es.get<SiPixelFedCablingMapRcd>().get( cablingMapLabel, cablingMap ); //Tav
    cablingMap_ = cablingMap.product();
    fedIds   = cablingMap->fedIds();
    cabling_ = cablingMap->cablingTree();
    LogDebug("map version:")<< cabling_->version();
  }
  // initialize quality record or update if necessary
  if (qualityWatcher.check( es )&&useQuality) {
    // quality info for dead pixel modules or ROCs
    edm::ESHandle<SiPixelQuality> qualityInfo;
    es.get<SiPixelQualityRcd>().get( qualityInfo );
    badPixelInfo_ = qualityInfo.product();
    if (!badPixelInfo_) {
      edm::LogError("SiPixelQualityNotPresent")<<" Configured to use SiPixelQuality, but SiPixelQuality not present";
    }
  }

  // tracker geometry: to make sure numbering of DetId is consistent...
  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get(geom);
  geom_ = geom.product();

  edm::ESHandle<TrackerTopology> trackerTopologyHandle;
  es.get<TrackerTopologyRcd>().get(trackerTopologyHandle);
  ttopo_ = trackerTopologyHandle.product();

  if (regions_) {
    regions_->run(ev, es);
    LogDebug("SiPixelRawToCluster") << "region2unpack #feds: "<<regions_->nFEDs();
    LogDebug("SiPixelRawToCluster") << "region2unpack #modules (BPIX,EPIX,total): "<<regions_->nBarrelModules()<<" "<<regions_->nForwardModules()<<" "<<regions_->nModules();
  }

  edm::Handle<FEDRawDataCollection> buffers;
  ev.getByToken(tFEDRawDataCollection, buffers);
  return buffers.product();
}


// -----------------------------------------------------------------------------
void SiPixelRawToClusterHeterogeneous::produceCPU(edm::HeterogeneousEvent& ev, const edm::EventSetup& es)
{
  const auto buffers = initialize(ev.event(), es);

  // create product (digis & errors)
  auto collection = std::make_unique<edm::DetSetVector<PixelDigi>>();
  auto errorcollection = std::make_unique<edm::DetSetVector<SiPixelRawDataError>>();
  auto tkerror_detidcollection = std::make_unique<DetIdCollection>();
  auto usererror_detidcollection = std::make_unique<DetIdCollection>();
  auto disabled_channelcollection = std::make_unique< edmNew::DetSetVector<PixelFEDChannel>>();
  auto outputClusters = std::make_unique<SiPixelClusterCollectionNew>();
  // output->collection.reserve(8*1024);


  PixelDataFormatter formatter(cabling_.get(), usePhase1); // for phase 1 & 0
  formatter.setErrorStatus(includeErrors);
  if (useQuality) formatter.setQualityStatus(useQuality, badPixelInfo_);

  bool errorsInEvent = false;
  PixelDataFormatter::DetErrors nodeterrors;

  if (regions_) {
    formatter.setModulesToUnpack(regions_->modulesToUnpack());
  }

  for (auto aFed = fedIds.begin(); aFed != fedIds.end(); ++aFed) {
    int fedId = *aFed;

    if(!usePilotBlade && (fedId==40) ) continue; // skip pilot blade data

    if (regions_ && !regions_->mayUnpackFED(fedId)) continue;

    if(debug) LogDebug("SiPixelRawToCluster")<< " PRODUCE DIGI FOR FED: " <<  fedId;

    PixelDataFormatter::Errors errors;

    //get event data for this fed
    const FEDRawData& fedRawData = buffers->FEDData( fedId );

    //convert data to digi and strip off errors
    formatter.interpretRawData( errorsInEvent, fedId, fedRawData, *collection, errors);

    //pack errors into collection
    if(includeErrors) {
      typedef PixelDataFormatter::Errors::iterator IE;
      for (IE is = errors.begin(); is != errors.end(); is++) {
	uint32_t errordetid = is->first;
	if (errordetid==dummydetid) {           // errors given dummy detId must be sorted by Fed
	  nodeterrors.insert( nodeterrors.end(), errors[errordetid].begin(), errors[errordetid].end() );
	} else {
	  edm::DetSet<SiPixelRawDataError>& errorDetSet = errorcollection->find_or_insert(errordetid);
	  errorDetSet.data.insert(errorDetSet.data.end(), is->second.begin(), is->second.end());
	  // Fill detid of the detectors where there is error AND the error number is listed
	  // in the configurable error list in the job option cfi.
	  // Code needs to be here, because there can be a set of errors for each
	  // entry in the for loop over PixelDataFormatter::Errors

	  std::vector<PixelFEDChannel> disabledChannelsDetSet;

	  for (auto const& aPixelError : errorDetSet) {
	    // For the time being, we extend the error handling functionality with ErrorType 25
	    // In the future, we should sort out how the usage of tkerrorlist can be generalized
	    if (aPixelError.getType()==25) {
	      assert(aPixelError.getFedId()==fedId);
	      const sipixelobjects::PixelFEDCabling* fed = cabling_->fed(fedId);
	      if (fed) {
		cms_uint32_t linkId = formatter.linkId(aPixelError.getWord32());
		const sipixelobjects::PixelFEDLink* link = fed->link(linkId);
		if (link) {
		  // The "offline" 0..15 numbering is fixed by definition, also, the FrameConversion depends on it
		  // in contrast, the ROC-in-channel numbering is determined by hardware --> better to use the "offline" scheme
		  PixelFEDChannel ch = {fed->id(), linkId, 25, 0};
		  for (unsigned int iRoc=1; iRoc<=link->numberOfROCs(); iRoc++) {
		    const sipixelobjects::PixelROC * roc = link->roc(iRoc);
		    if (roc->idInDetUnit()<ch.roc_first) ch.roc_first=roc->idInDetUnit();
		    if (roc->idInDetUnit()>ch.roc_last) ch.roc_last=roc->idInDetUnit();
		  }
		  disabledChannelsDetSet.push_back(ch);
		}
	      }
	    } else {
	      // fill list of detIds to be turned off by tracking
	      if(!tkerrorlist.empty()) {
		auto it_find = std::find(tkerrorlist.begin(), tkerrorlist.end(), aPixelError.getType());
		if(it_find != tkerrorlist.end()){
		  tkerror_detidcollection->push_back(errordetid);
		}
	      }
	    }

	    // fill list of detIds with errors to be studied
	    if(!usererrorlist.empty()) {
	      auto it_find = std::find(usererrorlist.begin(), usererrorlist.end(), aPixelError.getType());
	      if(it_find != usererrorlist.end()){
		usererror_detidcollection->push_back(errordetid);
	      }
	    }

	  } // loop on DetSet of errors

	  if (!disabledChannelsDetSet.empty()) {
	    disabled_channelcollection->insert(errordetid, disabledChannelsDetSet.data(), disabledChannelsDetSet.size());
	  }

	} // if error assigned to a real DetId
      } // loop on errors in event for this FED
    } // if errors to be included in the event
  } // loop on FED data to be unpacked

  if(includeErrors) {
    edm::DetSet<SiPixelRawDataError>& errorDetSet = errorcollection->find_or_insert(dummydetid);
    errorDetSet.data = nodeterrors;
  }
  if (errorsInEvent) LogDebug("SiPixelRawToCluster") << "Error words were stored in this event";

  // clusterize, originally from SiPixelClusterProducer
  for(const auto detset: *collection) {
    const auto detId = DetId(detset.detId());

    std::vector<short> badChannels; // why do we need this?

    // Comment: At the moment the clusterizer depends on geometry
    // to access information as the pixel topology (number of columns
    // and rows in a detector module).
    // In the future the geometry service will be replaced with
    // a ES service.
    const GeomDetUnit      * geoUnit = geom_->idToDetUnit( detId );
    const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
    edmNew::DetSetVector<SiPixelCluster>::FastFiller spc(*outputClusters, detset.detId());
    clusterizer_.clusterizeDetUnit(detset, pixDet, ttopo_, badChannels, spc);
    if ( spc.empty() ) {
      spc.abort();
    }
  }
  outputClusters->shrink_to_fit();

  //send digis and errors back to framework
  ev.put(std::move(collection));
  if(includeErrors){
    ev.put(std::move(errorcollection));
    ev.put(std::move(tkerror_detidcollection));
    ev.put(std::move(usererror_detidcollection), "UserErrorModules");
    ev.put(std::move(disabled_channelcollection));
  }
  ev.put(std::move(outputClusters));
}

// -----------------------------------------------------------------------------
void SiPixelRawToClusterHeterogeneous::acquireGPUCuda(const edm::HeterogeneousEvent& ev, const edm::EventSetup& es, cuda::stream_t<>& cudaStream) {
  const auto buffers = initialize(ev.event(), es);

  edm::ESHandle<SiPixelFedCablingMapGPUWrapper> hgpuMap;
  es.get<CkfComponentsRecord>().get(hgpuMap);
  if(hgpuMap->hasQuality() != useQuality) {
    throw cms::Exception("LogicError") << "UseQuality of the module (" << useQuality<< ") differs the one from SiPixelFedCablingMapGPUWrapper. Please fix your configuration.";
  }
  // get the GPU product already here so that the async transfer can begin
  const auto *gpuMap = hgpuMap->getGPUProductAsync(cudaStream);

  edm::cuda::device::unique_ptr<unsigned char[]> modulesToUnpackRegional;
  const unsigned char *gpuModulesToUnpack;
  if (regions_) {
    modulesToUnpackRegional = hgpuMap->getModToUnpRegionalAsync(*(regions_->modulesToUnpack()), cudaStream);
    gpuModulesToUnpack = modulesToUnpackRegional.get();
  }
  else {
    gpuModulesToUnpack = hgpuMap->getModToUnpAllAsync(cudaStream);
  }


  edm::ESHandle<SiPixelGainCalibrationForHLTGPU> hgains;
  es.get<SiPixelGainCalibrationForHLTGPURcd>().get(hgains);

  errors_.clear();

  // GPU specific: Data extraction for RawToDigi GPU
  unsigned int wordCounterGPU = 0;
  unsigned int fedCounter = 0;
  bool errorsInEvent = false;

  // In CPU algorithm this loop is part of PixelDataFormatter::interpretRawData()
  ErrorChecker errorcheck;
  auto wordFedAppender = pixelgpudetails::SiPixelRawToClusterGPUKernel::WordFedAppender(cudaStream);
  for (auto aFed = fedIds.begin(); aFed != fedIds.end(); ++aFed) {
    int fedId = *aFed;

    if (!usePilotBlade && (fedId==40) ) continue; // skip pilot blade data
    if (regions_ && !regions_->mayUnpackFED(fedId)) continue;

    // for GPU
    // first 150 index stores the fedId and next 150 will store the
    // start index of word in that fed
    assert(fedId>=1200);
    fedCounter++;

    // get event data for this fed
    const FEDRawData& rawData = buffers->FEDData( fedId );

    // GPU specific
    int nWords = rawData.size()/sizeof(cms_uint64_t);
    if (nWords == 0) {
      continue;
    }

    // check CRC bit
    const cms_uint64_t* trailer = reinterpret_cast<const cms_uint64_t* >(rawData.data())+(nWords-1);
    if (not errorcheck.checkCRC(errorsInEvent, fedId, trailer, errors_)) {
      continue;
    }

    // check headers
    const cms_uint64_t* header = reinterpret_cast<const cms_uint64_t* >(rawData.data()); header--;
    bool moreHeaders = true;
    while (moreHeaders) {
      header++;
      bool headerStatus = errorcheck.checkHeader(errorsInEvent, fedId, header, errors_);
      moreHeaders = headerStatus;
    }

    // check trailers
    bool moreTrailers = true;
    trailer++;
    while (moreTrailers) {
      trailer--;
      bool trailerStatus = errorcheck.checkTrailer(errorsInEvent, fedId, nWords, trailer, errors_);
      moreTrailers = trailerStatus;
    }

    const cms_uint32_t * bw = (const cms_uint32_t *)(header+1);
    const cms_uint32_t * ew = (const cms_uint32_t *)(trailer);

    assert(0 == (ew-bw)%2);
    wordFedAppender.initializeWordFed(fedId, wordCounterGPU, bw, (ew-bw));
    wordCounterGPU+=(ew-bw);

  } // end of for loop

  gpuAlgo_.makeClustersAsync(gpuMap, gpuModulesToUnpack, hgains->getGPUProductAsync(cudaStream),
                             wordFedAppender,
                             wordCounterGPU, fedCounter, convertADCtoElectrons,
                             useQuality, includeErrors, enableTransfer_, debug, cudaStream);
}

void SiPixelRawToClusterHeterogeneous::produceGPUCuda(edm::HeterogeneousEvent& ev, const edm::EventSetup& es, cuda::stream_t<>& cudaStream) {
  auto output = std::make_unique<GPUProduct>(gpuAlgo_.getProduct());

  if(enableConversion_) {
    convertGPUtoCPU(ev.event(), output->nDigis, gpuAlgo_.getCPUData());
  }

  ev.put<Output>(std::move(output), heterogeneous::DisableTransfer{});
}

void SiPixelRawToClusterHeterogeneous::convertGPUtoCPU(edm::Event& ev,
                                                       unsigned int nDigis,
                                                       pixelgpudetails::SiPixelRawToClusterGPUKernel::CPUData digis_clusters_h) const {
  // TODO: add the transfers here as well?

  auto collection = std::make_unique<edm::DetSetVector<PixelDigi>>();
  auto errorcollection = std::make_unique<edm::DetSetVector<SiPixelRawDataError>>();
  auto tkerror_detidcollection = std::make_unique<DetIdCollection>();
  auto usererror_detidcollection = std::make_unique<DetIdCollection>();
  auto disabled_channelcollection = std::make_unique< edmNew::DetSetVector<PixelFEDChannel>>();
  auto outputClusters = std::make_unique<SiPixelClusterCollectionNew>();

  edm::DetSet<PixelDigi> * detDigis=nullptr;
  for (uint32_t i = 0; i < nDigis; i++) {
    if (digis_clusters_h.pdigi[i]==0) continue;
    detDigis = &collection->find_or_insert(digis_clusters_h.rawIdArr[i]);
    if ( (*detDigis).empty() ) (*detDigis).data.reserve(32); // avoid the first relocations
    break;
  }

  int32_t nclus=-1;
  std::vector<AccretionCluster> aclusters(1024);
  auto totCluseFilled=0;

  auto fillClusters = [&](uint32_t detId){
    if (nclus<0) return; // this in reality should never happen
    edmNew::DetSetVector<SiPixelCluster>::FastFiller spc(*outputClusters, detId);
    auto layer = (DetId(detId).subdetId()==1) ? ttopo_->pxbLayer(detId) : 0;
    auto clusterThreshold = (layer==1) ? 2000 : 4000;
    for (int32_t ic=0; ic<nclus+1;++ic) {
      auto const & acluster = aclusters[ic];
      if ( acluster.charge < clusterThreshold) continue;
      SiPixelCluster cluster(acluster.isize,acluster.adc, acluster.x,acluster.y, acluster.xmin,acluster.ymin);
      ++totCluseFilled;
      // std::cout << "putting in this cluster " << ic << " " << cluster.charge() << " " << cluster.pixelADC().size() << endl;
      // sort by row (x)
      spc.push_back( std::move(cluster) );
      std::push_heap(spc.begin(),spc.end(),[](SiPixelCluster const & cl1,SiPixelCluster const & cl2) { return cl1.minPixelRow() < cl2.minPixelRow();});
    }
    for (int32_t ic=0; ic<nclus+1;++ic) aclusters[ic].clear();
    nclus = -1;
    // sort by row (x)
    std::sort_heap(spc.begin(),spc.end(),[](SiPixelCluster const & cl1,SiPixelCluster const & cl2) { return cl1.minPixelRow() < cl2.minPixelRow();});
    if ( spc.empty() ) spc.abort();
  };

  for (uint32_t i = 0; i < nDigis; i++) {
    if (digis_clusters_h.pdigi[i]==0) continue;
    if (digis_clusters_h.clus[i]>9000) continue; // not in cluster
    assert(digis_clusters_h.rawIdArr[i] > 109999);
    if ( (*detDigis).detId() != digis_clusters_h.rawIdArr[i])
      {
        fillClusters((*detDigis).detId());
        assert(nclus==-1);
        detDigis = &collection->find_or_insert(digis_clusters_h.rawIdArr[i]);
        if ( (*detDigis).empty() )
          (*detDigis).data.reserve(32); // avoid the first relocations
        else { std::cout << "Problem det present twice in input! " << (*detDigis).detId() << std::endl; }
      }
    (*detDigis).data.emplace_back(digis_clusters_h.pdigi[i]);
    auto const & dig = (*detDigis).data.back();
    // fill clusters
    assert(digis_clusters_h.clus[i]>=0);
    assert(digis_clusters_h.clus[i]<1024);
    nclus = std::max(digis_clusters_h.clus[i],nclus);
    auto row = dig.row();
    auto col = dig.column();
    SiPixelCluster::PixelPos pix(row,col);
    aclusters[digis_clusters_h.clus[i]].add(pix, digis_clusters_h.adc[i]);
  }

  // fill final clusters
  fillClusters((*detDigis).detId());
  //std::cout << "filled " << totCluseFilled << " clusters" << std::endl;

  PixelDataFormatter formatter(cabling_.get(), usePhase1); // for phase 1 & 0
  auto errors = errors_; // make a copy
  PixelDataFormatter::DetErrors nodeterrors;

  auto size = digis_clusters_h.error->size();
  for (auto i = 0; i < size; i++) {
    pixelgpudetails::error_obj err = (*digis_clusters_h.error)[i];
    if (err.errorType != 0) {
      SiPixelRawDataError error(err.word, err.errorType, err.fedId + 1200);
      errors[err.rawId].push_back(error);
    }
  }

  // pack errors into collection
  if (includeErrors) {

    typedef PixelDataFormatter::Errors::iterator IE;
    for (IE is = errors.begin(); is != errors.end(); is++) {

      uint32_t errordetid = is->first;
      if (errordetid == dummydetid) {// errors given dummy detId must be sorted by Fed
        nodeterrors.insert( nodeterrors.end(), errors[errordetid].begin(), errors[errordetid].end() );
      }
      else {
        edm::DetSet<SiPixelRawDataError>& errorDetSet = errorcollection->find_or_insert(errordetid);
        errorDetSet.data.insert(errorDetSet.data.end(), is->second.begin(), is->second.end());
        // Fill detid of the detectors where there is error AND the error number is listed
        // in the configurable error list in the job option cfi.
        // Code needs to be here, because there can be a set of errors for each
        // entry in the for loop over PixelDataFormatter::Errors

        std::vector<PixelFEDChannel> disabledChannelsDetSet;

        for (auto const& aPixelError : errorDetSet) {
          // For the time being, we extend the error handling functionality with ErrorType 25
          // In the future, we should sort out how the usage of tkerrorlist can be generalized
          if (aPixelError.getType() == 25) {
            int fedId = aPixelError.getFedId();
            const sipixelobjects::PixelFEDCabling* fed = cabling_->fed(fedId);
            if (fed) {
              cms_uint32_t linkId = formatter.linkId(aPixelError.getWord32());
              const sipixelobjects::PixelFEDLink* link = fed->link(linkId);
              if (link) {
                // The "offline" 0..15 numbering is fixed by definition, also, the FrameConversion depends on it
                // in contrast, the ROC-in-channel numbering is determined by hardware --> better to use the "offline" scheme
                PixelFEDChannel ch = {fed->id(), linkId, 25, 0};
                for (unsigned int iRoc = 1; iRoc <= link->numberOfROCs(); iRoc++) {
                  const sipixelobjects::PixelROC * roc = link->roc(iRoc);
                  if (roc->idInDetUnit() < ch.roc_first) ch.roc_first = roc->idInDetUnit();
                  if (roc->idInDetUnit() > ch.roc_last) ch.roc_last = roc->idInDetUnit();
                }
                if (ch.roc_first<ch.roc_last) disabledChannelsDetSet.push_back(ch);
              }
            }
          }
          else {
            // fill list of detIds to be turned off by tracking
            if (!tkerrorlist.empty()) {
              auto it_find = std::find(tkerrorlist.begin(), tkerrorlist.end(), aPixelError.getType());
              if (it_find != tkerrorlist.end()) {
                tkerror_detidcollection->push_back(errordetid);
              }
            }
          }

          // fill list of detIds with errors to be studied
          if (!usererrorlist.empty()) {
            auto it_find = std::find(usererrorlist.begin(), usererrorlist.end(), aPixelError.getType());
            if (it_find != usererrorlist.end()) {
              usererror_detidcollection->push_back(errordetid);
            }
          }

        } // loop on DetSet of errors

        if (!disabledChannelsDetSet.empty()) {
          disabled_channelcollection->insert(errordetid, disabledChannelsDetSet.data(), disabledChannelsDetSet.size());
        }

      } // if error assigned to a real DetId
    } // loop on errors in event for this FED
  } // if errors to be included in the event

  if (includeErrors) {
    edm::DetSet<SiPixelRawDataError>& errorDetSet = errorcollection->find_or_insert(dummydetid);
    errorDetSet.data = nodeterrors;
  }

  ev.put(std::move(collection));
  if(includeErrors){
    ev.put(std::move(errorcollection));
    ev.put(std::move(tkerror_detidcollection));
    ev.put(std::move(usererror_detidcollection), "UserErrorModules");
    ev.put(std::move(disabled_channelcollection));
  }
  ev.put(std::move(outputClusters));
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelRawToClusterHeterogeneous);
