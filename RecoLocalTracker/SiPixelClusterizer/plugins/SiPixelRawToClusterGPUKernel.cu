/* Sushil Dubey, Shashi Dugad, TIFR, July 2017
 *
 * File Name: RawToClusterGPU.cu
 * Description: It converts Raw data into Digi Format on GPU
 * then it converts adc -> electron and
 * applies the adc threshold to needed for clustering
 * Finaly the Output of RawToDigi data is given to pixelClusterizer
 *
**/

// C++ includes
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

// cub includes
#include <cub/cub.cuh>

// CMSSW includes
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuCalibPixel.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClustering.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClusterChargeCut.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelFedCablingMapGPU.h"

// local includes
#include "SiPixelRawToClusterGPUKernel.h"

namespace pixelgpudetails {

  // data structures size
  constexpr uint32_t vsize = sizeof(GPU::SimpleVector<pixelgpudetails::error_obj>);
  constexpr uint32_t esize = sizeof(pixelgpudetails::error_obj);

  // number of words for all the FEDs
  constexpr uint32_t MAX_FED_WORDS   = pixelgpudetails::MAX_FED * pixelgpudetails::MAX_WORD;
  constexpr uint32_t MAX_ERROR_SIZE  = MAX_FED_WORDS * esize;

  SiPixelRawToClusterGPUKernel::WordFedAppender::WordFedAppender(cuda::stream_t<>& cudaStream) {
    edm::Service<CUDAService> cs;
    word_ = cs->make_host_unique<unsigned int[]>(MAX_FED_WORDS, cudaStream);
    fedId_ = cs->make_host_unique<unsigned char[]>(MAX_FED_WORDS, cudaStream);
  }

  void SiPixelRawToClusterGPUKernel::WordFedAppender::initializeWordFed(int fedId, unsigned int wordCounterGPU, const cms_uint32_t *src, unsigned int length) {
    std::memcpy(word_.get()+wordCounterGPU, src, sizeof(cms_uint32_t)*length);
    std::memset(fedId_.get()+wordCounterGPU/2, fedId - 1200, length/2);
  }

  ////////////////////

  __device__ uint32_t getLink(uint32_t ww)  {
    return ((ww >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask);
  }


  __device__ uint32_t getRoc(uint32_t ww) {
    return ((ww >> pixelgpudetails::ROC_shift ) & pixelgpudetails::ROC_mask);
  }


  __device__ uint32_t getADC(uint32_t ww) {
    return ((ww >> pixelgpudetails::ADC_shift) & pixelgpudetails::ADC_mask);
  }


  __device__ bool isBarrel(uint32_t rawId) {
    return (1==((rawId>>25)&0x7));
  }

  __device__ pixelgpudetails::DetIdGPU getRawId(const SiPixelFedCablingMapGPU * Map, uint32_t fed, uint32_t link, uint32_t roc) {
    uint32_t index = fed * MAX_LINK * MAX_ROC + (link-1) * MAX_ROC + roc;
    pixelgpudetails::DetIdGPU detId = { Map->RawId[index], Map->rocInDet[index], Map->moduleId[index] };
    return detId;
  }

  //reference http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_9_2_0/doc/html/dd/d31/FrameConversion_8cc_source.html
  //http://cmslxr.fnal.gov/source/CondFormats/SiPixelObjects/src/PixelROC.cc?v=CMSSW_9_2_0#0071
  // Convert local pixel to pixelgpudetails::global pixel
  __device__ pixelgpudetails::Pixel frameConversion(bool bpix, int side, uint32_t layer, uint32_t rocIdInDetUnit, pixelgpudetails::Pixel local) {

    int slopeRow  = 0, slopeCol = 0;
    int rowOffset = 0, colOffset = 0;

    if (bpix) {

      if (side == -1 && layer != 1) { // -Z side: 4 non-flipped modules oriented like 'dddd', except Layer 1
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8-rocIdInDetUnit)*pixelgpudetails::numColsInRoc-1;
        }
        else {
          slopeRow  = -1;
          slopeCol  = 1;
          rowOffset = 2*pixelgpudetails::numRowsInRoc-1;
          colOffset = (rocIdInDetUnit-8)*pixelgpudetails::numColsInRoc;
        } // if roc
      }
      else { // +Z side: 4 non-flipped modules oriented like 'pppp', but all 8 in layer1
        if (rocIdInDetUnit < 8) {
          slopeRow  = -1;
          slopeCol  =  1;
          rowOffset = 2*pixelgpudetails::numRowsInRoc-1;
          colOffset = rocIdInDetUnit * pixelgpudetails::numColsInRoc;
        }
        else {
          slopeRow  = 1;
          slopeCol  = -1;
          rowOffset = 0;
          colOffset = (16-rocIdInDetUnit)*pixelgpudetails::numColsInRoc-1;
        }
      }

    }
    else { // fpix
      if (side==-1) { // pannel 1
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8-rocIdInDetUnit)*pixelgpudetails::numColsInRoc-1;
        }
        else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2*pixelgpudetails::numRowsInRoc-1;
          colOffset = (rocIdInDetUnit-8)*pixelgpudetails::numColsInRoc;
        }
      }
      else { // pannel 2
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8-rocIdInDetUnit)*pixelgpudetails::numColsInRoc-1;
        }
        else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2*pixelgpudetails::numRowsInRoc-1;
          colOffset = (rocIdInDetUnit-8)*pixelgpudetails::numColsInRoc;
        }

      } // side

    }

    uint32_t gRow = rowOffset+slopeRow*local.row;
    uint32_t gCol = colOffset+slopeCol*local.col;
    //printf("Inside frameConversion row: %u, column: %u\n",gRow, gCol);
    pixelgpudetails::Pixel global = {gRow, gCol};
    return global;
  }


  __device__ uint32_t conversionError(uint32_t fedId, uint32_t status, bool debug = false)
  {
    uint32_t errorType = 0;

    // debug = true;

    switch (status) {
      case(1) : {
        if (debug) printf("Error in Fed: %i, invalid channel Id (errorType = 35\n)", fedId );
        errorType = 35;
        break;
      }
      case(2) : {
        if (debug) printf("Error in Fed: %i, invalid ROC Id (errorType = 36)\n", fedId);
        errorType = 36;
        break;
      }
      case(3) : {
        if (debug) printf("Error in Fed: %i, invalid dcol/pixel value (errorType = 37)\n", fedId);
        errorType = 37;
        break;
      }
      case(4) : {
        if (debug) printf("Error in Fed: %i, dcol/pixel read out of order (errorType = 38)\n", fedId);
        errorType = 38;
        break;
      }
      default:
        if (debug) printf("Cabling check returned unexpected result, status = %i\n", status);
    };

    return errorType;
  }

  __device__ bool rocRowColIsValid(uint32_t rocRow, uint32_t rocCol)
  {
    uint32_t numRowsInRoc = 80;
    uint32_t numColsInRoc = 52;

    /// row and collumn in ROC representation
    return ((rocRow < numRowsInRoc) & (rocCol < numColsInRoc));
  }

  __device__ bool dcolIsValid(uint32_t dcol, uint32_t pxid)
  {
    return ((dcol < 26) &  (2 <= pxid) & (pxid < 162));
  }

  __device__ uint32_t checkROC(uint32_t errorWord, uint32_t fedId, uint32_t link, const SiPixelFedCablingMapGPU *Map, bool debug = false)
  {
    int errorType = (errorWord >> pixelgpudetails::ROC_shift) & pixelgpudetails::ERROR_mask;
    if (errorType < 25) return false;
    bool errorFound = false;

    switch (errorType) {
      case(25) : {
        errorFound = true;
        uint32_t index = fedId * MAX_LINK * MAX_ROC + (link-1) * MAX_ROC + 1;
        if (index > 1 && index <= Map->size) {
          if (!(link == Map->link[index] && 1 == Map->roc[index])) errorFound = false;
        }
        if (debug&errorFound) printf("Invalid ROC = 25 found (errorType = 25)\n");
        break;
      }
      case(26) : {
        if (debug) printf("Gap word found (errorType = 26)\n");
        errorFound = true;
        break;
      }
      case(27) : {
        if (debug) printf("Dummy word found (errorType = 27)\n");
        errorFound = true;
        break;
      }
      case(28) : {
        if (debug) printf("Error fifo nearly full (errorType = 28)\n");
        errorFound = true;
        break;
      }
      case(29) : {
        if (debug) printf("Timeout on a channel (errorType = 29)\n");
        if ((errorWord >> pixelgpudetails::OMIT_ERR_shift) & pixelgpudetails::OMIT_ERR_mask) {
          if (debug) printf("...first errorType=29 error, this gets masked out\n");
        }
        errorFound = true;
        break;
      }
      case(30) : {
        if (debug) printf("TBM error trailer (errorType = 30)\n");
        int StateMatch_bits = 4;
        int StateMatch_shift = 8;
        uint32_t StateMatch_mask = ~(~uint32_t(0) << StateMatch_bits);
        int StateMatch = (errorWord >> StateMatch_shift) & StateMatch_mask;
        if ( StateMatch != 1 && StateMatch != 8 ) {
          if (debug) printf("FED error 30 with unexpected State Bits (errorType = 30)\n");
        }
        if ( StateMatch == 1 ) errorType = 40; // 1=Overflow -> 40, 8=number of ROCs -> 30
        errorFound = true;
        break;
      }
      case(31) : {
        if (debug) printf("Event number error (errorType = 31)\n");
        errorFound = true;
        break;
      }
      default:
        errorFound = false;
    };

    return errorFound? errorType : 0;
  }

  __device__ uint32_t getErrRawID(uint32_t fedId, uint32_t errWord, uint32_t errorType, const SiPixelFedCablingMapGPU *Map, bool debug = false)
  {
    uint32_t rID = 0xffffffff;

    switch (errorType) {
      case 25 : case 30 : case 31 : case 36 : case 40 : {
        //set dummy values for cabling just to get detId from link
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        uint32_t roc  = 1;
        uint32_t link = (errWord >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask;
        uint32_t rID_temp = getRawId(Map, fedId, link, roc).RawId;
        if (rID_temp != 9999) rID = rID_temp;
        break;
      }
      case 29 : {
        int chanNmbr = 0;
        const int DB0_shift = 0;
        const int DB1_shift = DB0_shift + 1;
        const int DB2_shift = DB1_shift + 1;
        const int DB3_shift = DB2_shift + 1;
        const int DB4_shift = DB3_shift + 1;
        const uint32_t DataBit_mask = ~(~uint32_t(0) << 1);

        int CH1 = (errWord >> DB0_shift) & DataBit_mask;
        int CH2 = (errWord >> DB1_shift) & DataBit_mask;
        int CH3 = (errWord >> DB2_shift) & DataBit_mask;
        int CH4 = (errWord >> DB3_shift) & DataBit_mask;
        int CH5 = (errWord >> DB4_shift) & DataBit_mask;
        int BLOCK_bits      = 3;
        int BLOCK_shift     = 8;
        uint32_t BLOCK_mask = ~(~uint32_t(0) << BLOCK_bits);
        int BLOCK = (errWord >> BLOCK_shift) & BLOCK_mask;
        int localCH = 1*CH1+2*CH2+3*CH3+4*CH4+5*CH5;
        if (BLOCK%2==0) chanNmbr=(BLOCK/2)*9+localCH;
        else chanNmbr = ((BLOCK-1)/2)*9+4+localCH;
        if ((chanNmbr < 1)||(chanNmbr > 36)) break;  // signifies unexpected result

        // set dummy values for cabling just to get detId from link if in Barrel
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        uint32_t roc  = 1;
        uint32_t link = chanNmbr;
        uint32_t rID_temp = getRawId(Map, fedId, link, roc).RawId;
        if(rID_temp != 9999) rID = rID_temp;
        break;
      }
      case 37 : case 38: {
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        uint32_t roc  = (errWord >> pixelgpudetails::ROC_shift) & pixelgpudetails::ROC_mask;
        uint32_t link = (errWord >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask;
        uint32_t rID_temp = getRawId(Map, fedId, link, roc).RawId;
        if(rID_temp != 9999) rID = rID_temp;
        break;
      }
      default:
        break;
    };

    return rID;
  }

  /*----------
   * Name: applyADCthreshold_kernel()
   * Desc: converts adc count to electrons and then applies the
   * threshold on each channel.
   * make pixel to 0 if it is below the threshold
   * Input: xx_d[], yy_d[], layer_d[], wordCounter, adc[], ADCThreshold
   *-----------
   * Output: xx_adc[], yy_adc[] with pixel threshold applied
   */
  // kernel to apply adc threshold on the channels


  // Felice: gains and pedestals are not the same for each pixel. This code should be rewritten to take
  // in account local gains/pedestals
  // __global__ void applyADCthreshold_kernel(const uint32_t *xx_d, const uint32_t *yy_d, const uint32_t *layer_d, uint32_t *adc, const uint32_t wordCounter,
  //  const ADCThreshold adcThreshold, uint32_t *xx_adc, uint32_t *yy_adc ) {
  //   int tid = threadIdx.x;
  //   int gIndex = blockDim.x*blockIdx.x+tid;
  //   if (gIndex<wordCounter) {
  //     uint32_t adcOld = adc[gIndex];
  //     const float gain = adcThreshold.theElectronPerADCGain_; // default: 1 ADC = 135 electrons
  //     const float pedestal = 0; //
  //     int adcNew = int(adcOld*gain+pedestal);
  //     // rare chance of entering into the if ()
  //     if (layer_d[gIndex]>=adcThreshold.theFirstStack_) {
  //       if (adcThreshold.theStackADC_==1 && adcOld==1) {
  //         adcNew = int(255*135); // Arbitrarily use overflow value.
  //       }
  //       if (adcThreshold.theStackADC_ >1 && adcThreshold.theStackADC_!=255 && adcOld>=1){
  //         adcNew = int((adcOld-1) * gain * 255/float(adcThreshold.theStackADC_-1));
  //       }
  //     }
  //
  //     if (adcNew >adcThreshold.thePixelThreshold ) {
  //       xx_adc[gIndex]=xx_d[gIndex];
  //       yy_adc[gIndex]=yy_d[gIndex];
  //     }
  //     else {
  //       xx_adc[gIndex]=0; // 0: dead pixel
  //       yy_adc[gIndex]=0;
  //     }
  //     adc[gIndex] = adcNew;
  //   }
  // }


  // Kernel to perform Raw to Digi conversion
  __global__ void RawToDigi_kernel(const SiPixelFedCablingMapGPU *Map, const unsigned char *modToUnp,
      const uint32_t wordCounter, const uint32_t *Word, const uint8_t *fedIds,
      uint16_t * XX, uint16_t * YY, uint16_t * ADC,
      uint32_t * pdigi, uint32_t *rawIdArr, uint16_t * moduleId,
      GPU::SimpleVector<pixelgpudetails::error_obj> *err,
      bool useQualityInfo, bool includeErrors, bool debug)
  {
    uint32_t blockId  = blockIdx.x;
    uint32_t threadId  = threadIdx.x;

    bool skipROC = false;
    //if (threadId==0) printf("Event: %u blockId: %u start: %u end: %u\n", eventno, blockId, begin, end);

    for (int aaa=0; aaa<1; ++aaa) {  // too many coninue below.... (to be fixed)
      auto gIndex = threadId + blockId*blockDim.x;
      if (gIndex < wordCounter) {

        uint32_t fedId = fedIds[gIndex/2]; // +1200;

        // initialize (too many coninue below)
        pdigi[gIndex]  = 0;
        rawIdArr[gIndex] = 0;
        moduleId[gIndex] = 9999;

        uint32_t ww = Word[gIndex]; // Array containing 32 bit raw data
        if (ww == 0) {
          //noise and dead channels are ignored
          XX[gIndex]    = 0;  // 0 is an indicator of a noise/dead channel
          YY[gIndex]    = 0; // skip these pixels during clusterization
          ADC[gIndex]   = 0;
          continue; // 0: bad word
        }

        uint32_t link  = getLink(ww);            // Extract link
        uint32_t roc   = getRoc(ww);             // Extract Roc in link
        pixelgpudetails::DetIdGPU detId = getRawId(Map, fedId, link, roc);

        uint32_t errorType = checkROC(ww, fedId, link, Map, debug);
        skipROC = (roc < pixelgpudetails::maxROCIndex) ? false : (errorType != 0);
        if (includeErrors and skipROC)
        {
          uint32_t rID = getErrRawID(fedId, ww, errorType, Map, debug);
          err->emplace_back(rID, ww, errorType, fedId);
          continue;
        }

        uint32_t rawId  = detId.RawId;
        uint32_t rocIdInDetUnit = detId.rocInDet;
        bool barrel = isBarrel(rawId);

        uint32_t index = fedId * MAX_LINK * MAX_ROC + (link-1) * MAX_ROC + roc;
        if (useQualityInfo) {

          skipROC = Map->badRocs[index];
          if (skipROC) continue;

        }
        skipROC = modToUnp[index];
        if (skipROC) continue;

        uint32_t layer = 0;//, ladder =0;
        int side = 0, panel = 0, module = 0;//disk = 0,blade = 0

        if (barrel)
        {
          layer  = (rawId >> pixelgpudetails::layerStartBit) & pixelgpudetails::layerMask;
          module = (rawId >> pixelgpudetails::moduleStartBit) & pixelgpudetails::moduleMask;
          side   = (module < 5)? -1 : 1;
        }
        else {
          // endcap ids
          layer = 0;
          panel = (rawId >> pixelgpudetails::panelStartBit) & pixelgpudetails::panelMask;
          //disk  = (rawId >> diskStartBit_) & diskMask_;
          side  = (panel == 1)? -1 : 1;
          //blade = (rawId >> bladeStartBit_) & bladeMask_;
        }

        // ***special case of layer to 1 be handled here
        pixelgpudetails::Pixel localPix;
        if (layer == 1) {
          uint32_t col = (ww >> pixelgpudetails::COL_shift) & pixelgpudetails::COL_mask;
          uint32_t row = (ww >> pixelgpudetails::ROW_shift) & pixelgpudetails::ROW_mask;
          localPix.row = row;
          localPix.col = col;
          if (includeErrors) {
            if (not rocRowColIsValid(row, col)) {
              uint32_t error = conversionError(fedId, 3, debug); //use the device function and fill the arrays
              err->emplace_back(rawId, ww, error, fedId);
              if(debug) printf("BPIX1  Error status: %i\n", error);
              continue;
            }
          }
        } else {
          // ***conversion rules for dcol and pxid
          uint32_t dcol = (ww >> pixelgpudetails::DCOL_shift) & pixelgpudetails::DCOL_mask;
          uint32_t pxid = (ww >> pixelgpudetails::PXID_shift) & pixelgpudetails::PXID_mask;
          uint32_t row  = pixelgpudetails::numRowsInRoc - pxid/2;
          uint32_t col  = dcol*2 + pxid%2;
          localPix.row = row;
          localPix.col = col;
          if (includeErrors and not dcolIsValid(dcol, pxid)) {
            uint32_t error = conversionError(fedId, 3, debug);
            err->emplace_back(rawId, ww, error, fedId);
            if(debug) printf("Error status: %i %d %d %d %d\n", error, dcol, pxid, fedId, roc);
            continue;
          }
        }

        pixelgpudetails::Pixel globalPix = frameConversion(barrel, side, layer, rocIdInDetUnit, localPix);
        XX[gIndex]    = globalPix.row;  // origin shifting by 1 0-159
        YY[gIndex]    = globalPix.col;  // origin shifting by 1 0-415
        ADC[gIndex]   = getADC(ww);
        pdigi[gIndex] = pixelgpudetails::pack(globalPix.row,globalPix.col,ADC[gIndex]);
        moduleId[gIndex] = detId.moduleId;
        rawIdArr[gIndex] = rawId;
      } // end of if (gIndex < end)
    } // end fake loop
  } // end of Raw to Digi kernel

  // Interface to outside
  void SiPixelRawToClusterGPUKernel::makeClustersAsync(
      const SiPixelFedCablingMapGPU *cablingMap,
      const unsigned char *modToUnp,
      const SiPixelGainForHLTonGPU *gains,
      const WordFedAppender& wordFed,
      const uint32_t wordCounter, const uint32_t fedCounter,
      bool convertADCtoElectrons,
      bool useQualityInfo, bool includeErrors, bool transferToCPU, bool debug,
      cuda::stream_t<>& stream)
  {
    nDigis = wordCounter;

    constexpr uint32_t MAX_FED_WORDS   = pixelgpudetails::MAX_FED * pixelgpudetails::MAX_WORD;
    digis_d = SiPixelDigisCUDA(MAX_FED_WORDS, stream);
    clusters_d = SiPixelClustersCUDA(MAX_FED_WORDS, gpuClustering::MaxNumModules, stream);

    edm::Service<CUDAService> cs;
    digis_clusters_h.nModules_Clusters = cs->make_host_unique<uint32_t[]>(2, stream);

    {
      const int threadsPerBlock = 512;
      const int blocks = (wordCounter + threadsPerBlock-1) /threadsPerBlock; // fill it all

      assert(0 == wordCounter%2);
      // wordCounter is the total no of words in each event to be trasfered on device
      auto word_d = cs->make_device_unique<uint32_t[]>(wordCounter, stream);
      auto fedId_d = cs->make_device_unique<uint8_t[]>(wordCounter, stream);

      auto error_d = cs->make_device_unique<GPU::SimpleVector<pixelgpudetails::error_obj>>(stream);
      auto data_d = cs->make_device_unique<pixelgpudetails::error_obj[]>(MAX_FED_WORDS, stream);
      cudaCheck(cudaMemsetAsync(data_d.get(), 0x00, MAX_ERROR_SIZE, stream.id()));
      auto error_h_tmp = cs->make_host_unique<GPU::SimpleVector<pixelgpudetails::error_obj>>(stream);
      new (error_h_tmp.get()) GPU::SimpleVector<pixelgpudetails::error_obj>(MAX_FED_WORDS, data_d.get()); // should make_host_unique() call the constructor as well? note that even if std::make_unique does that, we can't do that in make_device_unique
      assert(error_h_tmp->size() == 0);
      assert(error_h_tmp->capacity() == static_cast<int>(MAX_FED_WORDS));

      cudaCheck(cudaMemcpyAsync(word_d.get(),  wordFed.word(), wordCounter*sizeof(uint32_t),    cudaMemcpyDefault, stream.id()));
      cudaCheck(cudaMemcpyAsync(fedId_d.get(), wordFed.fedId(), wordCounter*sizeof(uint8_t) / 2, cudaMemcpyDefault, stream.id()));
      cudaCheck(cudaMemcpyAsync(error_d.get(), error_h_tmp.get(), vsize, cudaMemcpyDefault, stream.id()));

      auto pdigi_d = cs->make_device_unique<uint32_t[]>(wordCounter, stream);
      auto rawIdArr_d = cs->make_device_unique<uint32_t[]>(wordCounter, stream);

      // Launch rawToDigi kernel
      RawToDigi_kernel<<<blocks, threadsPerBlock, 0, stream.id()>>>(
          cablingMap,
          modToUnp,
          wordCounter,
          word_d.get(),
          fedId_d.get(),
          digis_d.xx(), digis_d.yy(), digis_d.adc(),
          pdigi_d.get(),
          rawIdArr_d.get(),
          digis_d.moduleInd(),
          error_d.get(),
          useQualityInfo,
          includeErrors,
          debug);
      cudaCheck(cudaGetLastError());

      // copy data to host variable
      if(transferToCPU) {
        digis_clusters_h.pdigi = cs->make_host_unique<uint32_t[]>(MAX_FED_WORDS, stream);
        digis_clusters_h.rawIdArr = cs->make_host_unique<uint32_t[]>(MAX_FED_WORDS, stream);
        cudaCheck(cudaMemcpyAsync(digis_clusters_h.pdigi.get(), pdigi_d.get(), wordCounter*sizeof(uint32_t), cudaMemcpyDefault, stream.id()));
        cudaCheck(cudaMemcpyAsync(digis_clusters_h.rawIdArr.get(), rawIdArr_d.get(), wordCounter*sizeof(uint32_t), cudaMemcpyDefault, stream.id()));

        if (includeErrors) {
          digis_clusters_h.data = cs->make_host_unique<pixelgpudetails::error_obj[]>(MAX_FED_WORDS, stream);
          digis_clusters_h.error = cs->make_host_unique<GPU::SimpleVector<pixelgpudetails::error_obj>>(stream);
          new (digis_clusters_h.error.get()) GPU::SimpleVector<pixelgpudetails::error_obj>(MAX_FED_WORDS, digis_clusters_h.data.get());
          assert(digis_clusters_h.error->size() == 0);
          assert(digis_clusters_h.error->capacity() == static_cast<int>(MAX_FED_WORDS));

          cudaCheck(cudaMemcpyAsync(digis_clusters_h.error.get(), error_d.get(), vsize, cudaMemcpyDefault, stream.id()));
          cudaCheck(cudaMemcpyAsync(digis_clusters_h.data.get(), data_d.get(), MAX_ERROR_SIZE, cudaMemcpyDefault, stream.id()));
          // If we want to transfer only the minimal amount of data, we
          // need a synchronization point. A single ExternalWork (of
          // SiPixelRawToClusterHeterogeneous) does not help because it is
          // already used to synchronize the data movement. So we'd need
          // two ExternalWorks (or explicit use of TBB tasks). The
          // prototype of #100 would allow this easily (as there would be
          // two ExternalWorks).
          //
          //cudaCheck(cudaStreamSynchronize(stream.id()));
          //int size = digis_clusters_h.error->size();
          //cudaCheck(cudaMemcpyAsync(digis_clusters_h.data.get(), data_d.get(), size*esize, cudaMemcpyDefault, stream.id()));
        }
      }
    }
    // End  of Raw2Digi and passing data for cluserisation

    {
      // clusterizer ...
      using namespace gpuClustering;
      int threadsPerBlock = 256;
      int blocks = (wordCounter + threadsPerBlock - 1) / threadsPerBlock;

      gpuCalibPixel::calibDigis<<<blocks, threadsPerBlock, 0, stream.id()>>>(
          digis_d.moduleInd(),
          digis_d.c_xx(), digis_d.c_yy(), digis_d.adc(),
          gains,
          wordCounter);
      cudaCheck(cudaGetLastError());

      // calibrated adc
      if(transferToCPU) {
        digis_clusters_h.adc = cs->make_host_unique<uint16_t[]>(MAX_FED_WORDS, stream);
        cudaCheck(cudaMemcpyAsync(digis_clusters_h.adc.get(), digis_d.adc(), wordCounter*sizeof(uint16_t), cudaMemcpyDefault, stream.id()));
      }

#ifdef GPU_DEBUG
       std::cout
         << "CUDA countModules kernel launch with " << blocks
         << " blocks of " << threadsPerBlock << " threads\n";
#endif

      cudaCheck(cudaMemsetAsync(clusters_d.moduleStart(), 0x00, sizeof(uint32_t), stream.id()));

      countModules<<<blocks, threadsPerBlock, 0, stream.id()>>>(digis_d.c_moduleInd(), clusters_d.moduleStart(), clusters_d.clus(), wordCounter);
      cudaCheck(cudaGetLastError());

      // read the number of modules into a data member, used by getProduct())
      cudaCheck(cudaMemcpyAsync(&(digis_clusters_h.nModules_Clusters[0]), clusters_d.moduleStart(), sizeof(uint32_t), cudaMemcpyDefault, stream.id()));

      threadsPerBlock = 256;
      blocks = MaxNumModules;
#ifdef GPU_DEBUG
         std::cout << "CUDA findClus kernel launch with " << blocks
         << " blocks of " << threadsPerBlock << " threads\n";
#endif
      cudaCheck(cudaMemsetAsync(clusters_d.clusInModule(), 0, (MaxNumModules)*sizeof(uint32_t), stream.id()));
      findClus<<<blocks, threadsPerBlock, 0, stream.id()>>>(
          digis_d.c_moduleInd(),
          digis_d.c_xx(), digis_d.c_yy(),
          clusters_d.c_moduleStart(),
          clusters_d.clusInModule(), clusters_d.moduleId(),
          clusters_d.clus(),
          wordCounter);
      cudaCheck(cudaGetLastError());

      // apply charge cut
      clusterChargeCut<<<blocks, threadsPerBlock, 0, stream.id()>>>(
          digis_d.moduleInd(),
          digis_d.c_adc(),
          clusters_d.c_moduleStart(),
          clusters_d.clusInModule(), clusters_d.c_moduleId(),
          clusters_d.clus(),
          wordCounter);
      cudaCheck(cudaGetLastError());



      // count the module start indices already here (instead of
      // rechits) so that the number of clusters/hits can be made
      // available in the rechit producer without additional points of
      // synchronization/ExternalWork
      //
      // Temporary storage
      size_t tempScanStorageSize = 0;
      {
        uint32_t *tmp = nullptr;
        cudaCheck(cub::DeviceScan::InclusiveSum(nullptr, tempScanStorageSize, tmp, tmp, MaxNumModules));
      }
      auto tempScanStorage_d = cs->make_device_unique<uint32_t[]>(tempScanStorageSize, stream);
      // Set first the first element to 0
      cudaCheck(cudaMemsetAsync(clusters_d.clusModuleStart(), 0, sizeof(uint32_t), stream.id()));
      // Then use inclusive_scan to get the partial sum to the rest
      cudaCheck(cub::DeviceScan::InclusiveSum(tempScanStorage_d.get(), tempScanStorageSize,
                                              clusters_d.c_clusInModule(), &clusters_d.clusModuleStart()[1], gpuClustering::MaxNumModules,
                                              stream.id()));
      // last element holds the number of all clusters
      cudaCheck(cudaMemcpyAsync(&(digis_clusters_h.nModules_Clusters[1]), clusters_d.clusModuleStart()+gpuClustering::MaxNumModules, sizeof(uint32_t), cudaMemcpyDefault, stream.id()));


      // clusters
      if(transferToCPU) {
        digis_clusters_h.clus = cs->make_host_unique<int32_t[]>(MAX_FED_WORDS, stream);
        cudaCheck(cudaMemcpyAsync(digis_clusters_h.clus.get(), clusters_d.clus(), wordCounter*sizeof(uint32_t), cudaMemcpyDefault, stream.id()));
      }
    } // end clusterizer scope
  }

}
