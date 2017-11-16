// Skip FED40 pilot-blade
// Include parameter driven interface to SiPixelQuality for study purposes
// exclude ROC(raw) based on bad ROC list in SiPixelQuality
// enabled by: process.siPixelDigis.UseQualityInfo = True (BY DEFAULT NOT USED)
// 20-10-2010 Andrew York (Tennessee)
// Jan 2016 Tamas Almos Vami (Tav) (Wigner RCP) -- Cabling Map label option
// Jul 2017 Viktor Veszpremi -- added PixelFEDChannel

// This code is an entry point for GPU based pixel track reconstruction for HLT
// Modified by Sushil and Shashi for this purpose July-2017

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelUnpackingRegions.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "TH1D.h"
#include "TFile.h"
#include "SiPixelRawToDigiGPU.h"
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>

#include <cuda.h>             // for GPU
#include <cuda_runtime.h>     // for pinned memory
#include "EventInfoGPU.h"     // Event Info
// device memory intialization for RawTodigi
#include "RawToDigiMem.h"
// // device memory initialization for CPE
// #include "CPEGPUMem.h"
// //device memory initialization for clustering
// #include "PixelClusterMem.h"

using namespace std;

// -----------------------------------------------------------------------------
SiPixelRawToDigiGPU::SiPixelRawToDigiGPU( const edm::ParameterSet& conf ) 
  : config_(conf), 
    badPixelInfo_(nullptr),
    regions_(nullptr),
    hCPU(nullptr), hDigi(nullptr)
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

  //start counters
  ndigis = 0;
  nwords = 0;

  // Products
  produces< edm::DetSetVector<PixelDigi> >();
  if(includeErrors){
    produces< edm::DetSetVector<SiPixelRawDataError> >();
    produces<DetIdCollection>();
    produces<DetIdCollection>("UserErrorModules");
    produces<edmNew::DetSetVector<PixelFEDChannel> >();
  }

  // regions
  if (config_.exists("Regions")) {
    if(!config_.getParameter<edm::ParameterSet>("Regions").getParameterNames().empty())
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
    if(usePilotBlade) edm::LogInfo("SiPixelRawToDigiGPU")  << " Use pilot blade data (FED 40)";
  }

  // Control the usage of phase1
  usePhase1 = false;
  if (config_.exists("UsePhase1")) {
    usePhase1 = config_.getParameter<bool> ("UsePhase1");
    if(usePhase1) edm::LogInfo("SiPixelRawToDigiGPU")  << " Using phase1";
  }
  //CablingMap could have a label //Tav
  cablingMapLabel = config_.getParameter<std::string> ("CablingMapLabel");
  
  //GPU specific
  convertADCtoElectrons = config_.getParameter<bool>("ConvertADCtoElectrons");
  const int MAX_FED  = 150;
  const int MAX_WORD = 2000;
  int WSIZE    = MAX_FED*MAX_WORD*NEVENT*sizeof(unsigned int);
  int FSIZE    = 2*MAX_FED*NEVENT*sizeof(unsigned int)+sizeof(unsigned int); 

  //word = (unsigned int*)malloc(WSIZE);
  cudaMallocHost((void**)&word, WSIZE);
  //fedIndex =(unsigned int*)malloc(FSIZE);
  cudaMallocHost((void**)&fedIndex, FSIZE);
  eventIndex = (unsigned int*)malloc((NEVENT+1)*sizeof(unsigned int));
  eventIndex[0] =0;
  // to store the output of RawToDigi
  xx_h = new uint[WSIZE];
  yy_h = new uint[WSIZE];
  adc_h = new uint[WSIZE];

  mIndexStart_h = new int[NEVENT*NMODULE +1];
  mIndexEnd_h = new int[NEVENT*NMODULE +1];

  // allocate memory for RawToDigi on GPU
  initDeviceMemory();

  // // allocate auxilary memory for clustering
  // initDeviceMemCluster();

  // // allocate memory for CPE on GPU
  // initDeviceMemCPE();
  
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
  // free(word);
  cudaFreeHost(word);
  // free(fedIndex);
  cudaFreeHost(fedIndex);
  free(eventIndex);
  delete[] xx_h;
  delete[] yy_h;
  delete[] adc_h;
  delete[] mIndexStart_h;
  delete[] mIndexEnd_h;
  // free device memory used for RawToDigi on GPU
  freeMemory(); 
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
  descriptions.add("siPixelRawToDigi",desc);
}

// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
void SiPixelRawToDigiGPU::produce( edm::Event& ev,
                              const edm::EventSetup& es) 
{
  //const uint32_t dummydetid = 0xffffffff;
  //debug = edm::MessageDrop::instance()->debugEnabled;

  // initialize cabling map or update if necessary
  if (recordWatcher.check( es )) {
    // cabling map, which maps online address (fed->link->ROC->local pixel) to offline (DetId->global pixel)
    edm::ESTransientHandle<SiPixelFedCablingMap> cablingMap;
    es.get<SiPixelFedCablingMapRcd>().get( cablingMapLabel, cablingMap ); //Tav
    fedIds   = cablingMap->fedIds();
    // cabling_ = cablingMap->cablingTree();
    // LogDebug("map version:")<< cabling_->version();
  }

/*  // initialize quality record or update if necessary
  if (qualityWatcher.check( es )&&useQuality) {
    // quality info for dead pixel modules or ROCs
    edm::ESHandle<SiPixelQuality> qualityInfo;
    es.get<SiPixelQualityRcd>().get( qualityInfo );
    badPixelInfo_ = qualityInfo.product();
    if (!badPixelInfo_) {
      edm::LogError("SiPixelQualityNotPresent")<<" Configured to use SiPixelQuality, but SiPixelQuality not present"<<endl;
    }
  }*/

  edm::Handle<FEDRawDataCollection> buffers;
  ev.getByToken(tFEDRawDataCollection, buffers);

  /*// create product (digis & errors)
  auto collection = std::make_unique<edm::DetSetVector<PixelDigi>>();
  // collection->reserve(8*1024);
  auto errorcollection = std::make_unique<edm::DetSetVector<SiPixelRawDataError>>();
  auto tkerror_detidcollection = std::make_unique<DetIdCollection>();
  auto usererror_detidcollection = std::make_unique<DetIdCollection>();
  auto disabled_channelcollection = std::make_unique<edmNew::DetSetVector<PixelFEDChannel> >();

  //PixelDataFormatter formatter(cabling_.get()); // phase 0 only
  PixelDataFormatter formatter(cabling_.get(), usePhase1); // for phase 1 & 0

  formatter.setErrorStatus(includeErrors);

  if (useQuality) formatter.setQualityStatus(useQuality, badPixelInfo_);

  if (theTimer) theTimer->start();
  bool errorsInEvent = false;
  PixelDataFormatter::DetErrors nodeterrors;

  if (regions_) {
    regions_->run(ev, es);
    formatter.setModulesToUnpack(regions_->modulesToUnpack());
    LogDebug("SiPixelRawToDigiGPU") << "region2unpack #feds: "<<regions_->nFEDs();
    LogDebug("SiPixelRawToDigiGPU") << "region2unpack #modules (BPIX,EPIX,total): "<<regions_->nBarrelModules()<<" "<<regions_->nForwardModules()<<" "<<regions_->nModules();
  }*/
  // GPU specific: Data extraction for RawToDigi GPU
  static unsigned int wordCounterGPU =0;
  unsigned int fedCounter = 0;
  const unsigned int MAX_FED = 150;
  static int eventCount = 0;
  bool errorsInEvent = false;
  
  ErrorChecker errorcheck;
  for (auto aFed = fedIds.begin(); aFed != fedIds.end(); ++aFed) {
    int fedId = *aFed;
   
    // for GPU
    // first 150 index stores the fedId and next 150 will store the
    // start index of word in that fed
    fedIndex[2*MAX_FED*eventCount+fedCounter] = fedId-1200;
    fedIndex[MAX_FED + 2*MAX_FED*eventCount+fedCounter] = wordCounterGPU; // MAX_FED = 150
    fedCounter++;

    //get event data for this fed
    const FEDRawData& rawData = buffers->FEDData( fedId );
    //GPU specific 
    PixelDataFormatter::Errors errors;
    int nWords = rawData.size()/sizeof(Word64);
    if(nWords==0) {
      word[wordCounterGPU++] =0;
      continue;
    }  

    // check CRC bit
    const Word64* trailer = reinterpret_cast<const Word64* >(rawData.data())+(nWords-1);  
    if(!errorcheck.checkCRC(errorsInEvent, fedId, trailer, errors)) {
      word[wordCounterGPU++] =0;
      continue;
    }

    // check headers
    const Word64* header = reinterpret_cast<const Word64* >(rawData.data()); header--;
    bool moreHeaders = true;
    while (moreHeaders) {
      header++;
      //LogTrace("")<<"HEADER:  " <<  print(*header);
      bool headerStatus = errorcheck.checkHeader(errorsInEvent, fedId, header, errors);
      moreHeaders = headerStatus;
    }

    // check trailers
    bool moreTrailers = true;
    trailer++;
    while (moreTrailers) {
      trailer--;
      //LogTrace("")<<"TRAILER: " <<  print(*trailer);
      bool trailerStatus = errorcheck.checkTrailer(errorsInEvent, fedId, nWords, trailer, errors);
      moreTrailers = trailerStatus;
    }

    const  Word32 * bw =(const  Word32 *)(header+1);
    const  Word32 * ew =(const  Word32 *)(trailer);
    if ( *(ew-1) == 0 ) { ew--; }
    for (auto ww = bw; ww < ew; ++ww) {
      word[wordCounterGPU++] = *ww;
    }
  }  // end of for loop
  
  // GPU specific: RawToDigi -> clustering -> CPE
  eventCount++;
  eventIndex[eventCount] = wordCounterGPU;
  static int ec=1;
  cout<<"Data read for event: "<<ec++<<endl;
  int r2d_debug=0;
  if(eventCount==NEVENT) {
    RawToDigi_wrapper(wordCounterGPU, word, fedCounter,fedIndex, eventIndex, convertADCtoElectrons, xx_h, yy_h, adc_h, mIndexStart_h, mIndexEnd_h);

    if(r2d_debug==1){
      //write output to text file (for debugging purpose only)
      static int count = 1;
      cout << "Writing output to the file " << endl;
      ofstream ofileXY("GPU_RawToDigi_Output_Part1_Event_"+to_string((count-1)*NEVENT+1)+"to"+to_string(count*NEVENT)+".txt");
      ofileXY<<"  Index     xcor   ycor    adc  "<<endl;
      for(uint i=0; i<wordCounterGPU; i++) {
        ofileXY<<setw(10) << i  << setw(6) << xx_h[i] << setw(6) << yy_h[i] << setw(8) << adc_h[i] << endl;
      }
      //store the module index, which stores the index of x, y & adc for each module
      ofstream ofileModule("GPU_RawToDigi_Output_Part2_Event_"+to_string((count-1)*NEVENT+1)+"to"+to_string(count*NEVENT)+".txt");
      ofileModule<<"Event_No  Module_no   mIndexStart   mIndexEnd "<<endl;
      for(int ev = (count-1)*NEVENT+1; ev <=count*NEVENT; ev++) {
        for(int mod =0; mod < NMODULE; mod++) {
          ofileModule << setw(8) << ev << setw(8) <<mod << setw(10) << mIndexStart_h[mod] << setw(10) << mIndexEnd_h[mod]<<endl;
        }
      }
      count++;
      ofileXY.close();
      ofileModule.close();
    }
    wordCounterGPU =  0;
    eventCount=0;
  } //if(eventCount == NEVENT)
  fedCounter =0;
} //end of produce function

//define as runnable module
DEFINE_FWK_MODULE(SiPixelRawToDigiGPU);
