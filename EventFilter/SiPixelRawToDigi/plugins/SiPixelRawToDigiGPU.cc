// This code is an entry point for GPU based pixel track reconstruction for HLT
// Modified by Sushil and Shashi for this purpose July-2017

#include <string>
#include <chrono>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <TH1D.h>
#include <TFile.h>

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"



#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelUnpackingRegions.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventInfoGPU.h"
#include "RawToDigiGPU.h"
#include "SiPixelFedCablingMapGPU.h"
#include "SiPixelRawToDigiGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"


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
}


using namespace std;

// -----------------------------------------------------------------------------
SiPixelRawToDigiGPU::SiPixelRawToDigiGPU( const edm::ParameterSet& conf )
  : config_(conf),
    badPixelInfo_(nullptr),
    regions_(nullptr),
    hCPU(nullptr), hDigi(nullptr),
    theSiPixelGainCalibration_(conf)
{
  includeErrors = config_.getParameter<bool>("IncludeErrors");
  useQuality = config_.getParameter<bool>("UseQualityInfo");
  if (config_.exists("ErrorList")) {
    tkerrorlist = config_.getParameter<std::vector<int> > ("ErrorList");
  }
  if (config_.exists("UserErrorList")) {
    usererrorlist = config_.getParameter<std::vector<int> > ("UserErrorList");
  }
  tFEDRawDataCollection = consumes <FEDRawDataCollection> (config_.getParameter<edm::InputTag>("InputLabel"));
  debug = config_.getParameter<bool>("enableErrorDebug");

  //start counters
  ndigis = 0;
  nwords = 0;

  // Products
  produces< edm::DetSetVector<PixelDigi> >();
  produces<SiPixelClusterCollectionNew>(); 
  if (includeErrors) {
    produces< edm::DetSetVector<SiPixelRawDataError> >();
    produces<DetIdCollection>();
    produces<DetIdCollection>("UserErrorModules");
    produces<edmNew::DetSetVector<PixelFEDChannel> >();
  }

  // GPU "product"
  produces<std::vector<unsigned long long>>();

  // regions
  if (config_.exists("Regions")) {
    if (!config_.getParameter<edm::ParameterSet>("Regions").getParameterNames().empty())
    {
      regions_ = new PixelUnpackingRegions(config_, consumesCollector());
    }
  }

  // Timing
  bool timing = config_.getUntrackedParameter<bool>("Timing",false);
  if (timing) {
    theTimer.reset( new edm::CPUTimer );
    hCPU = new TH1D ("hCPU","hCPU",100,0.,0.050);
    hDigi = new TH1D("hDigi","hDigi",50,0.,15000.);
  }

  // Control the usage of pilot-blade data, FED=40
  usePilotBlade = false;
  if (config_.exists("UsePilotBlade")) {
    usePilotBlade = config_.getParameter<bool> ("UsePilotBlade");
    if (usePilotBlade) edm::LogInfo("SiPixelRawToDigiGPU")  << " Use pilot blade data (FED 40)";
  }

  // Control the usage of phase1
  usePhase1 = false;
  if (config_.exists("UsePhase1")) {
    usePhase1 = config_.getParameter<bool> ("UsePhase1");
    if (usePhase1) edm::LogInfo("SiPixelRawToDigiGPU")  << " Using phase1";
  }
  // CablingMap could have a label //Tav
  cablingMapLabel = config_.getParameter<std::string> ("CablingMapLabel");

  // GPU specific
  convertADCtoElectrons = config_.getParameter<bool>("ConvertADCtoElectrons");

  // device copy of GPU friendly cablng map
  allocateCablingMap(cablingMapGPUHost_, cablingMapGPUDevice_);

  int WSIZE = MAX_FED * MAX_WORD * sizeof(unsigned int);
  cudaMallocHost(&word,       sizeof(unsigned int)*WSIZE);
  cudaMallocHost(&fedId_h,    sizeof(unsigned char)*WSIZE);

  // to store the output of RawToDigi
  cudaMallocHost(&pdigi_h,    sizeof(uint32_t)*WSIZE);
  cudaMallocHost(&rawIdArr_h, sizeof(uint32_t)*WSIZE);

  cudaMallocHost(&adc_h, sizeof(uint16_t)*WSIZE);
  cudaMallocHost(&clus_h, sizeof(int32_t)*WSIZE);

  uint32_t vsize = sizeof(GPU::SimpleVector<error_obj>);
  uint32_t esize = sizeof(error_obj);
  cudaCheck(cudaMallocHost(&error_h, vsize));
  cudaCheck(cudaMallocHost(&error_h_tmp, vsize));
  cudaCheck(cudaMallocHost(&data_h, MAX_FED*MAX_WORD*esize));

  cudaMallocHost(&mIndexStart_h, sizeof(int)*(NMODULE+1));
  cudaMallocHost(&mIndexEnd_h,   sizeof(int)*(NMODULE+1));

  // allocate memory for RawToDigi on GPU
  context_ = initDeviceMemory();

  new (error_h) GPU::SimpleVector<error_obj>(MAX_FED*MAX_WORD, data_h);
  new (error_h_tmp) GPU::SimpleVector<error_obj>(MAX_FED*MAX_WORD, context_.data_d);
  assert(error_h->size() == 0);
  assert(error_h->capacity() == static_cast<int>(MAX_FED*MAX_WORD));
  assert(error_h_tmp->size() == 0);
  assert(error_h_tmp->capacity() == static_cast<int>(MAX_FED*MAX_WORD));
}

// -----------------------------------------------------------------------------
SiPixelRawToDigiGPU::~SiPixelRawToDigiGPU() {
  edm::LogInfo("SiPixelRawToDigiGPU")  << " HERE ** SiPixelRawToDigiGPU destructor!";

  if (regions_) delete regions_;

  if (theTimer) {
    TFile rootFile("analysis.root", "RECREATE", "my histograms");
    hCPU->Write();
    hDigi->Write();
  }
  cudaFreeHost(word);
  cudaFreeHost(fedId_h);
  cudaFreeHost(pdigi_h);
  cudaFreeHost(rawIdArr_h);
  cudaFreeHost(adc_h);
  cudaFreeHost(clus_h);
  cudaFreeHost(error_h);
  cudaFreeHost(error_h_tmp);
  cudaFreeHost(data_h);
  cudaFreeHost(mIndexStart_h);
  cudaFreeHost(mIndexEnd_h);

  // release device memory for cabling map
  deallocateCablingMap(cablingMapGPUHost_, cablingMapGPUDevice_);

  // free gains device memory
  cudaCheck(cudaFree(gainForHLTonGPU_));
  cudaCheck(cudaFree(gainDataOnGPU_));

  // free device memory used for RawToDigi on GPU
  freeMemory(context_);

  // free auxilary memory used for clustering
  // freeDeviceMemCluster();
  // free device memory used for CPE on GPU
  // freeDeviceMemCPE();
}

void
SiPixelRawToDigiGPU::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
  desc.add<edm::InputTag>("InputLabel",edm::InputTag("siPixelRawData"));
  {
    edm::ParameterSetDescription psd0;
    psd0.addOptional<std::vector<edm::InputTag>>("inputs");
    psd0.addOptional<std::vector<double>>("deltaPhi");
    psd0.addOptional<std::vector<double>>("maxZ");
    psd0.addOptional<edm::InputTag>("beamSpot");
    desc.add<edm::ParameterSetDescription>("Regions",psd0)->setComment("## Empty Regions PSet means complete unpacking");
  }
  desc.addUntracked<bool>("Timing",false);
  desc.add<bool>("UsePilotBlade",false)->setComment("##  Use pilot blades");
  desc.add<bool>("UsePhase1",false)->setComment("##  Use phase1");
  desc.add<std::string>("CablingMapLabel","")->setComment("CablingMap label"); //Tav
  desc.addOptional<bool>("CheckPixelOrder");  // never used, kept for back-compatibility
  desc.add<bool>("ConvertADCtoElectrons", false)->setComment("## do the calibration ADC-> Electron and apply the threshold, requried for clustering");
  desc.add<bool>("enableErrorDebug",false);
  descriptions.add("siPixelRawToDigiGPU",desc);
}

// -----------------------------------------------------------------------------
void
SiPixelRawToDigiGPU::produce( edm::Event& ev, const edm::EventSetup& es)
{
  // setup gain calibration service
  theSiPixelGainCalibration_.setESObjects( es );

   edm::ESHandle<TrackerTopology> trackerTopologyHandle;
   es.get<TrackerTopologyRcd>().get(trackerTopologyHandle);
   tTopo_ = trackerTopologyHandle.product();


  int theWordCounter = 0;
  int theDigiCounter = 0;
  const uint32_t dummydetid = 0xffffffff;

  // initialize quality record or update if necessary
  if (qualityWatcher.check( es ) && useQuality) {
    // quality info for dead pixel modules or ROCs
    edm::ESHandle<SiPixelQuality> qualityInfo;
    es.get<SiPixelQualityRcd>().get( qualityInfo );
    badPixelInfo_ = qualityInfo.product();
    if (!badPixelInfo_) {
      edm::LogError("SiPixelQualityNotPresent") << " Configured to use SiPixelQuality, but SiPixelQuality not present" << endl;
    }
  }

  std::set<unsigned int> modules;
  if (regions_) {
    regions_->run(ev, es);
    LogDebug("SiPixelRawToDigiGPU") << "region2unpack #feds: " << regions_->nFEDs();
    LogDebug("SiPixelRawToDigiGPU") << "region2unpack #modules (BPIX,EPIX,total): " << regions_->nBarrelModules() << " " << regions_->nForwardModules() << " " << regions_->nModules();
    modules = *(regions_->modulesToUnpack());
  }

  // initialize cabling map or update if necessary
  if (recordWatcher.check( es )) {
    // tracker geometry: to make sure numbering of DetId is consistent...
    edm::ESHandle<TrackerGeometry> geom;
    // get the TrackerGeom
    es.get<TrackerDigiGeometryRecord>().get( geom );

    // cabling map, which maps online address (fed->link->ROC->local pixel) to offline (DetId->global pixel)
    edm::ESTransientHandle<SiPixelFedCablingMap> cablingMap;
    es.get<SiPixelFedCablingMapRcd>().get( cablingMapLabel, cablingMap ); //Tav
    fedIds   = cablingMap->fedIds();
    cabling_ = cablingMap->cablingTree();
    LogDebug("map version:") << cabling_->version();

    // convert the cabling map to a GPU-friendly version
    processCablingMap(*cablingMap, *geom.product(), cablingMapGPUHost_, cablingMapGPUDevice_, badPixelInfo_, modules);
    processGainCalibration(theSiPixelGainCalibration_.payload(), *geom.product(), gainForHLTonGPU_, gainDataOnGPU_);
  }

  edm::Handle<FEDRawDataCollection> buffers;
  ev.getByToken(tFEDRawDataCollection, buffers);

  // create product (digis & errors)
  auto collection = std::make_unique<edm::DetSetVector<PixelDigi>>();
  // collection->reserve(8*1024);
  auto errorcollection = std::make_unique<edm::DetSetVector<SiPixelRawDataError>>();
  auto tkerror_detidcollection = std::make_unique<DetIdCollection>();
  auto usererror_detidcollection = std::make_unique<DetIdCollection>();
  auto disabled_channelcollection = std::make_unique<edmNew::DetSetVector<PixelFEDChannel> >();

  // create product (clusters);
  auto outputClusters = std::make_unique< SiPixelClusterCollectionNew>();


  //PixelDataFormatter formatter(cabling_.get()); // phase 0 only
  PixelDataFormatter formatter(cabling_.get(), usePhase1); // for phase 1 & 0
  PixelDataFormatter::DetErrors nodeterrors;
  PixelDataFormatter::Errors errors;

  if (theTimer) theTimer->start();

  // GPU specific: Data extraction for RawToDigi GPU
  unsigned int wordCounterGPU = 0;
  unsigned int fedCounter = 0;
  bool errorsInEvent = false;

  edm::DetSet<PixelDigi> * detDigis=nullptr;

  ErrorChecker errorcheck;
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
    if (not errorcheck.checkCRC(errorsInEvent, fedId, trailer, errors)) {
      continue;
    }

    // check headers
    const cms_uint64_t* header = reinterpret_cast<const cms_uint64_t* >(rawData.data()); header--;
    bool moreHeaders = true;
    while (moreHeaders) {
      header++;
      bool headerStatus = errorcheck.checkHeader(errorsInEvent, fedId, header, errors);
      moreHeaders = headerStatus;
    }

    // check trailers
    bool moreTrailers = true;
    trailer++;
    while (moreTrailers) {
      trailer--;
      bool trailerStatus = errorcheck.checkTrailer(errorsInEvent, fedId, nWords, trailer, errors);
      moreTrailers = trailerStatus;
    }

    theWordCounter += 2*(nWords-2);

    const cms_uint32_t * bw = (const cms_uint32_t *)(header+1);
    const cms_uint32_t * ew = (const cms_uint32_t *)(trailer);

    assert(0 == (ew-bw)%2);
    std::memcpy(word+wordCounterGPU,bw,sizeof(cms_uint32_t)*(ew-bw));
    std::memset(fedId_h+wordCounterGPU/2,fedId - 1200,(ew-bw)/2);
    wordCounterGPU+=(ew-bw);

  } // end of for loop

  // GPU specific: RawToDigi -> clustering
  uint32_t nModulesActive=0;
  RawToDigi_wrapper(context_, cablingMapGPUDevice_, gainForHLTonGPU_, wordCounterGPU, word, fedCounter,
                    fedId_h, convertADCtoElectrons, pdigi_h, rawIdArr_h, error_h, error_h_tmp, data_h,
                    adc_h, clus_h,
                    useQuality, includeErrors, debug, nModulesActive);

  auto gpuProd = std::make_unique<std::vector<unsigned long long>>();
  gpuProd->resize(3);
  (*gpuProd)[0]=uint64_t(&context_);
  (*gpuProd)[1]=wordCounterGPU;
  (*gpuProd)[2]=nModulesActive;
  ev.put(std::move(gpuProd));

  for (uint32_t i = 0; i < wordCounterGPU; i++) {
    if (pdigi_h[i]==0) continue;
    detDigis = &(*collection).find_or_insert(rawIdArr_h[i]);
    if ( (*detDigis).empty() ) (*detDigis).data.reserve(32); // avoid the first relocations
    break;
  }

  int32_t nclus=-1;
  std::vector<AccretionCluster> aclusters(256);
  auto totCluseFilled=0;

  auto fillClusters = [&](uint32_t detId){
    if (nclus<0) return; // this in reality should never happen
    edmNew::DetSetVector<SiPixelCluster>::FastFiller spc(*outputClusters, detId);
    auto layer = (DetId(detId).subdetId()==1) ? tTopo_->pxbLayer(detId) : 0;
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
    nclus=-1;                                         
    // sort by row (x)
    std::sort_heap(spc.begin(),spc.end(),[](SiPixelCluster const & cl1,SiPixelCluster const & cl2) { return cl1.minPixelRow() < cl2.minPixelRow();});
    if ( spc.empty() ) spc.abort();
   };

  for (uint32_t i = 0; i < wordCounterGPU; i++) {
    if (pdigi_h[i]==0) continue;
    assert(rawIdArr_h[i] > 109999);
    if ( (*detDigis).detId() != rawIdArr_h[i])
    {
      fillClusters((*detDigis).detId());
      assert(nclus==-1);
      detDigis = &(*collection).find_or_insert(rawIdArr_h[i]);
      if ( (*detDigis).empty() )
        (*detDigis).data.reserve(32); // avoid the first relocations
      else { std::cout << "Problem det present twice in input! " << (*detDigis).detId() << std::endl; }
    }
    (*detDigis).data.emplace_back(pdigi_h[i]);
    auto const & dig = (*detDigis).data.back();
    // fill clusters
    assert(clus_h[i]>=0);
    assert(clus_h[i]<256);
    nclus = std::max(clus_h[i],nclus);
    auto row = dig.row();
    auto col = dig.column();
    SiPixelCluster::PixelPos pix(row,col);
    aclusters[clus_h[i]].add(pix,adc_h[i]);
    theDigiCounter++;
  }

  // fill final clusters
  fillClusters((*detDigis).detId());
  //std::cout << "filled " << totCluseFilled << " clusters" << std::endl;

  auto size = error_h->size();
  for (auto i = 0; i < size; i++) {
    error_obj err = (*error_h)[i];
    if (err.errorType != 0) {
        SiPixelRawDataError error(err.word, err.errorType, err.fedId + 1200);
        errors[err.rawId].push_back(error);
    }
  }

  if (theTimer) {
    theTimer->stop();
    LogDebug("SiPixelRawToDigiGPU") << "TIMING IS: (real)" << theTimer->realTime() ;
    ndigis += theDigiCounter;
    nwords += theWordCounter;
    LogDebug("SiPixelRawToDigiGPU") << " (Words/Digis) this ev: "
      << theWordCounter << "/" << theDigiCounter << "--- all :" << nwords << "/" << ndigis;
    hCPU->Fill(theTimer->realTime());
    hDigi->Fill(theDigiCounter);
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
              std::vector<int>::iterator it_find = find(tkerrorlist.begin(), tkerrorlist.end(), aPixelError.getType());
              if (it_find != tkerrorlist.end()) {
                tkerror_detidcollection->push_back(errordetid);
              }
            }
          }

          // fill list of detIds with errors to be studied
          if (!usererrorlist.empty()) {
            std::vector<int>::iterator it_find = find(usererrorlist.begin(), usererrorlist.end(), aPixelError.getType());
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

  // send digis clusters and errors back to framework
  // std::cout << "Number of Clusters from GPU to CPU " << (*outputClusters).data().size() << std::endl;
  ev.put(std::move(collection));
  ev.put(std::move(outputClusters));
  if (includeErrors) {
    ev.put(std::move(errorcollection));
    ev.put(std::move(tkerror_detidcollection));
    ev.put(std::move(usererror_detidcollection), "UserErrorModules");
    ev.put(std::move(disabled_channelcollection));
  }

} // end of produce function

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelRawToDigiGPU);
