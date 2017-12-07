/* Sushil Dubey, Shashi Dugad, TIFR, July 2017
 *
 * File Name: RawToDigiGPU.cu
 * Description: It converts Raw data into Digi Format on GPU
 * then it converts adc -> electron and 
 * applies the adc threshold to needed for clustering
 * Finaly the Output of RawToDigi data is given to pixelClusterizer
 *
**/ 

// System includes
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <assert.h>
#include <iomanip>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include "CudaError.h"
#include "EventInfoGPU.h"
#include "RawToDigiGPU.h"
#include "RawToDigiMem.h"
#include "CablingMapGPU.h"

using namespace std;

// // forward declaration of pixelCluster_wrapper()
// void PixelCluster_Wrapper(uint *xx_adc, uint *yy_adc, uint *adc_d,const uint wordCounter, 
//                           const int *mIndexStart, const int *mIndexEnd);

/*
  This functions checks for cuda error
  Input: debug message
  Output: returns cuda error message
*/
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
    exit(-1);
  }
}


void initDeviceMemory() {
  // int sizeByte = MAX_FED * MAX_LINK * MAX_ROC * sizeof(uint)+sizeof(uint);
  // // Unified memory for cabling map
  // cudaMallocManaged((void**)&Map, sizeof(CablingMap));
  // cudaMallocManaged((void**)&Map->RawId,    sizeByte);
  // cudaMallocManaged((void**)&Map->rocInDet, sizeByte);
  // cudaMallocManaged((void**)&Map->moduleId, sizeByte);
  // Number of words for all the feds 
  uint MAX_WORD_SIZE = MAX_FED*MAX_WORD*NEVENT*sizeof(uint); 
  uint FSIZE = 2*MAX_FED*NEVENT*sizeof(uint)+sizeof(uint);
  
  int MSIZE = NMODULE*NEVENT*sizeof(int)+sizeof(int);

  cudaMalloc((void**)&eventIndex_d, (NEVENT+1)*sizeof(uint));

  cudaMalloc((void**)&word_d,       MAX_WORD_SIZE);
  cudaMalloc((void**)&fedIndex_d,   FSIZE);
  cudaMalloc((void**)&xx_d,         MAX_WORD_SIZE); // to store the x and y coordinate
  cudaMalloc((void**)&yy_d,         MAX_WORD_SIZE);
  cudaMalloc((void**)&xx_adc,       MAX_WORD_SIZE); // to store the x and y coordinate
  cudaMalloc((void**)&yy_adc,       MAX_WORD_SIZE);
  cudaMalloc((void**)&adc_d,        MAX_WORD_SIZE);
  cudaMalloc((void**)&layer_d ,     MAX_WORD_SIZE);
  cudaMalloc((void**)&rawIdArr_d,   MAX_WORD_SIZE);
    
  cudaMalloc((void**)&errType_d,   MAX_WORD_SIZE);
  cudaMalloc((void**)&errWord_d,   MAX_WORD_SIZE);
  cudaMalloc((void**)&errFedID_d,   MAX_WORD_SIZE);
  cudaMalloc((void**)&errRawID_d,   MAX_WORD_SIZE);

  cudaMalloc((void**)&moduleId_d,   MAX_WORD_SIZE);
  cudaMalloc((void**)&mIndexStart_d, MSIZE);
  cudaMalloc((void**)&mIndexEnd_d,   MSIZE);
  // create stream for RawToDigi 
  for(int i=0;i<NSTREAM;i++) {
    cudaStreamCreate(&stream[i]);
  }
  
  cout<<"Memory Allocated successfully !\n";
  // Upload the cabling Map
  // initCablingMap();
  
}


void freeMemory() {

  //GPU specific
  cudaFree(eventIndex_d);
  cudaFree(word_d);
  cudaFree(fedIndex_d);
  cudaFree(adc_d);
  cudaFree(layer_d);
  cudaFree(xx_d);
  cudaFree(yy_d);
  cudaFree(xx_adc);
  cudaFree(yy_adc);
  cudaFree(rawIdArr_d);
  
  cudaFree(moduleId_d);
  cudaFree(mIndexStart_d);
  cudaFree(mIndexEnd_d);

  // cudaFree(Map->RawId);
  // cudaFree(Map->rocInDet); 
  // cudaFree(Map->moduleId);
  // cudaFree(Map);

  // destroy the stream
  for(int i=0;i<NSTREAM;i++) {
    cudaStreamDestroy(stream[i]);
  }
  cout<<"Memory Released !\n";

}


__device__ uint getLink(uint ww)  {
  //printf("Link_shift: %d  LINK_mask: %d\n", LINK_shift, LINK_mask);
  return ((ww >> LINK_shift) & LINK_mask);
}


__device__ uint getRoc(uint ww) {
  return ((ww >> ROC_shift ) & ROC_mask);
}


__device__ uint getADC(uint ww) {
  return ((ww >> ADC_shift) & ADC_mask);
}


__device__ bool isBarrel(uint rawId) {
  return (1==((rawId>>25)&0x7));
}


//__device__ uint FED_START = 1200;


__device__ DetIdGPU getRawId(const CablingMap *Map, uint fed, uint link, uint roc) {
  uint index = fed * MAX_LINK * MAX_ROC + (link-1) * MAX_ROC + roc;
  DetIdGPU detId = {Map->RawId[index], Map->rocInDet[index], Map->moduleId[index]};
  return detId;  
}


//reference http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_9_2_0/doc/html/dd/d31/FrameConversion_8cc_source.html
//http://cmslxr.fnal.gov/source/CondFormats/SiPixelObjects/src/PixelROC.cc?v=CMSSW_9_2_0#0071
// Convert local pixel to global pixel
__device__ Pixel frameConversion(bool bpix, int side, uint layer, uint rocIdInDetUnit, Pixel local) {
  
  int slopeRow  = 0,  slopeCol = 0;
  int rowOffset = 0, colOffset = 0;

  if(bpix) {
    
    if(side == -1 && layer != 1) { // -Z side: 4 non-flipped modules oriented like 'dddd', except Layer 1
      if (rocIdInDetUnit < 8) {
        slopeRow = 1;     
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (8-rocIdInDetUnit)*numColsInRoc-1;
      }
      else {
        slopeRow  = -1;
        slopeCol  = 1;
        rowOffset = 2*numRowsInRoc-1;
        colOffset = (rocIdInDetUnit-8)*numColsInRoc;
      } // if roc
    }
    else { // +Z side: 4 non-flipped modules oriented like 'pppp', but all 8 in layer1
      if(rocIdInDetUnit < 8) {
        slopeRow  = -1;
        slopeCol  =  1;
        rowOffset = 2*numRowsInRoc-1;
        colOffset = rocIdInDetUnit * numColsInRoc;
      }
      else {
        slopeRow  = 1;
        slopeCol  = -1;
        rowOffset = 0;
        colOffset = (16-rocIdInDetUnit)*numColsInRoc-1;
      }
    }

  }
  else { // fpix
    if(side==-1) { // pannel 1
      if (rocIdInDetUnit < 8) {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (8-rocIdInDetUnit)*numColsInRoc-1;
      }
      else {
        slopeRow = -1;
        slopeCol = 1;
        rowOffset = 2*numRowsInRoc-1;
        colOffset = (rocIdInDetUnit-8)*numColsInRoc;
      }
    }
    else { // pannel 2
      if (rocIdInDetUnit < 8) {
        slopeRow = 1;
        slopeCol = -1;
        rowOffset = 0;
        colOffset = (8-rocIdInDetUnit)*numColsInRoc-1;
      }
      else {
        slopeRow = -1;
        slopeCol = 1;
        rowOffset = 2*numRowsInRoc-1;
        colOffset = (rocIdInDetUnit-8)*numColsInRoc;
      }

    } // side

  }

  uint gRow = rowOffset+slopeRow*local.row;
  uint gCol = colOffset+slopeCol*local.col;
  //printf("Inside frameConversion row: %u, column: %u\n",gRow, gCol);
  Pixel global = {gRow, gCol};
  return global;
}


__device__ uint conversionError(uint fedId, uint status, bool debug = false)
{
    
  uint errorType = 0;
    
  switch (status) {
      case(1) : {
        if(debug) printf("Error in Fed: %i, invalid channel Id (errorType=35)", fedId );
        errorType = 35;
        break;
      }
      case(2) : {
        if(debug) printf("Error in Fed: %i, invalid ROC Id (errorType=36)", fedId);
        errorType = 36;
        break;
      }
      case(3) : {
        if(debug) printf("Error in Fed: %i, invalid dcol/pixel value (errorType=37)", fedId);
        errorType = 37;
        break;
      }
      case(4) : {
        if(debug) printf("Error in Fed: %i, dcol/pixel read out of order (errorType=38)", fedId);
        errorType = 38;
        break;
      }
      default: if(debug) printf("Cabling check returned unexpected result, status = %i", status);
  };
    
  return errorType;
    
}


__device__ bool rocRowColIsValid(uint rocRow, uint rocCol)
{
    uint numRowsInRoc = 80;
    uint numColsInRoc = 52;
    
    /// row and collumn in ROC representation
    return ( (rocRow < numRowsInRoc) & (rocCol < numColsInRoc) );
}


__device__ bool dcolIsValid(uint dcol, uint pxid)
{
    return ( (dcol < 26) &  (2 <= pxid) & (pxid < 162) );
}


__device__ uint checkROC(uint errorWord, uint fedId, uint link, const CablingMap *Map, bool debug = false)
{
    
 int errorType = (errorWord >> ROC_shift) & ERROR_mask;
 bool errorFound = false;

 switch (errorType) {
    case(25) : {
     uint index = fedId * MAX_LINK * MAX_ROC + (link-1) * MAX_ROC + 1;
     if(index > 1 && index <= Map->size){
       if(!(link == Map->link[index] && 1 == Map->roc[index])) errorFound = false;
     }
     else{
       errorFound = true;
       if(debug) printf("Invalid ROC = 25 found (errorType=25)");
     }
     break;
   }
   case(26) : {
     if(debug) printf("Gap word found (errorType=26)");
     errorFound = true;
     break;
   }
   case(27) : {
     if(debug) printf("Dummy word found (errorType=27)");
     errorFound = true;
     break;
   }
   case(28) : {
     if(debug) printf("Error fifo nearly full (errorType=28)");
     errorFound = true;
     break;
   }
   case(29) : {
     if(debug) printf("Timeout on a channel (errorType=29)");
     if ((errorWord >> OMIT_ERR_shift) & OMIT_ERR_mask) {
       if(debug) printf("...first errorType=29 error, this gets masked out");
     }
     errorFound = true;
     break;
   }
   case(30) : {
     if(debug) printf("TBM error trailer (errorType=30)");
     int StateMatch_bits      = 4;
     int StateMatch_shift     = 8;
     uint32_t StateMatch_mask = ~(~uint32_t(0) << StateMatch_bits);
     int StateMatch = (errorWord >> StateMatch_shift) & StateMatch_mask;
     if( StateMatch != 1 && StateMatch != 8 ) {
       if(debug) printf("FED error 30 with unexpected State Bits (errorType=30)");
     }
     if( StateMatch==1 ) errorType = 40; // 1=Overflow -> 40, 8=number of ROCs -> 30
     errorFound = true;
     break;
   }
   case(31) : {
     if(debug) printf("Event number error (errorType=31)");
     errorFound = true;
     break;
   }
   default: errorFound = false;

 };

 return errorFound? errorType : 0;
    
}


__device__ uint getErrRawID(uint fedId, uint errWord, uint errorType, const CablingMap *Map, bool debug = false)
{
    
  uint rID = 0xffffffff;

  switch (errorType) {
    case  25 : case  30 : case  31 : case  36 : case 40 : {
      // set dummy values for cabling just to get detId from link
      //cabling.dcol = 0;
      //cabling.pxid = 2;
      uint roc  = 1;
      uint link = (errWord >> LINK_shift) & LINK_mask;
      
      rID = getRawId(Map, fedId, link, roc).RawId;
      break;
    }
    case  29 : {
      int chanNmbr = 0;
      const int DB0_shift = 0;
      const int DB1_shift = DB0_shift + 1;
      const int DB2_shift = DB1_shift + 1;
      const int DB3_shift = DB2_shift + 1;
      const int DB4_shift = DB3_shift + 1;
      const uint DataBit_mask = ~(~uint(0) << 1);

      int CH1 = (errWord >> DB0_shift) & DataBit_mask;
      int CH2 = (errWord >> DB1_shift) & DataBit_mask;
      int CH3 = (errWord >> DB2_shift) & DataBit_mask;
      int CH4 = (errWord >> DB3_shift) & DataBit_mask;
      int CH5 = (errWord >> DB4_shift) & DataBit_mask;
      int BLOCK_bits      = 3;
      int BLOCK_shift     = 8;
      uint BLOCK_mask = ~(~uint(0) << BLOCK_bits);
      int BLOCK = (errWord >> BLOCK_shift) & BLOCK_mask;
      int localCH = 1*CH1+2*CH2+3*CH3+4*CH4+5*CH5;
      if (BLOCK%2==0) chanNmbr=(BLOCK/2)*9+localCH;
      else chanNmbr = ((BLOCK-1)/2)*9+4+localCH;
      if ((chanNmbr < 1)||(chanNmbr > 36)) break;  // signifies unexpected result
      
      // set dummy values for cabling just to get detId from link if in Barrel
      //cabling.dcol = 0;
      //cabling.pxid = 2;
      uint roc  = 1;
      uint link = chanNmbr;
      rID = getRawId(Map, fedId, link, roc).RawId;
      break;
    }
    case  37 : case  38: {
      //cabling.dcol = 0;
      //cabling.pxid = 2;
      uint roc  = (errWord >> ROC_shift) & ROC_mask;
      uint link = (errWord >> LINK_shift) & LINK_mask;
      rID = getRawId(Map, fedId, link, roc).RawId;
      break;
    }
          
  default : break;
          
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
__global__ void applyADCthreshold_kernel
(const uint *xx_d, const uint *yy_d, const uint *layer_d, uint *adc, const uint wordCounter,
 const ADCThreshold adcThreshold, uint *xx_adc, uint *yy_adc ) {
  int tid = threadIdx.x;
  int gIndex = blockDim.x*blockIdx.x+tid;
  if(gIndex<wordCounter) {
    uint adcOld = adc[gIndex];
    const float gain = adcThreshold.theElectronPerADCGain_; // default: 1 ADC = 135 electrons
    const float pedestal = 0; //
    int adcNew = int(adcOld*gain+pedestal);
    // rare chance of entering into the if()
    if (layer_d[gIndex]>=adcThreshold.theFirstStack_) {
      if (adcThreshold.theStackADC_==1 && adcOld==1) {
        adcNew = int(255*135); // Arbitrarily use overflow value.
      }
      if (adcThreshold.theStackADC_ >1 && adcThreshold.theStackADC_!=255 && adcOld>=1){
        adcNew = int((adcOld-1) * gain * 255/float(adcThreshold.theStackADC_-1));
      }
    }
  
    if(adcNew >adcThreshold.thePixelThreshold ) {
      xx_adc[gIndex]=xx_d[gIndex];
      yy_adc[gIndex]=yy_d[gIndex];
    }
    else {
      xx_adc[gIndex]=0; // 0: dead pixel
      yy_adc[gIndex]=0;
    }
    adc[gIndex] = adcNew;
  }
}  


// Kernel to perform Raw to Digi conversion
__global__ void RawToDigi_kernel(const CablingMap *Map, const uint *Word, const uint *fedIndex,
                                 uint *eventIndex, const uint stream, uint *XX, uint *YY, uint *moduleId, int *mIndexStart,
                                 int *mIndexEnd, uint *ADC, uint *layerArr, uint *rawIdArr,
                                 uint *errType, uint *errWord, uint *errFedID, uint *errRawID,
                                 bool useQualityInfo, bool includeErrors, bool debug)
{
  uint blockId  = blockIdx.x;
  uint eventno  = blockIdx.y + gridDim.y*stream;
  
  //const uint eventOffset  = eventIndex[eventno]; 
  uint fedOffset = 2*MAX_FED*eventno;
  uint fedId     = fedIndex[fedOffset + blockId];
  uint threadId  = threadIdx.x;
  
  uint begin  = fedIndex[fedOffset + MAX_FED + blockId];
  uint end    = fedIndex[fedOffset + MAX_FED + blockId + 1];

  if(blockIdx.x == gridDim.x - 1) {
    end = eventIndex[eventno + 1]; // for last fed to get the end index
  }

  uint link = 0;
  uint roc  = 0;

  bool skipROC = false;
  //if(threadId==0) printf("Event: %u blockId: %u start: %u end: %u\n", eventno, blockId, begin, end);
  int no_itr = (end - begin)/blockDim.x + 1; // to deal with number of hits greater than blockDim.x 
  #pragma unroll
  for(int i = 0; i < no_itr; i++) { // use a static number to optimize this loop
    uint gIndex = begin + threadId + i*blockDim.x; 
    if(gIndex < end) {
      uint ww = Word[gIndex]; // Array containing 32 bit raw data
      if(ww == 0) {
        //noise and dead channels are ignored
        XX[gIndex]    = 0;  // 0 is an indicator of a noise/dead channel
        YY[gIndex]    = 0; // skip these pixels during clusterization
        ADC[gIndex]   = 0;
        layerArr[gIndex] = 0; 
        moduleId[gIndex] = 9999; //9999 is the indication of bad module, taken care later
        rawIdArr[gIndex] = 9999;
        continue ; // 0: bad word,
      }
      
      uint nlink  = getLink(ww);            // Extract link
      uint nroc   = getRoc(ww);             // Extract Roc in link
      if (!((nlink != link) | (nroc != roc))) continue;
      link = nlink;
      roc = nroc;
        
      uint errorType = checkROC(ww, fedId, link, Map, debug);
      skipROC = (roc < maxROCIndex) ? false : (errorType != 0);
      if (skipROC && includeErrors) {
        uint rID = getErrRawID(fedId, ww, errorType, Map, debug); //write the function
        errType[gIndex] = errorType;
        errWord[gIndex] = ww;
        errFedID[gIndex] = fedId;
        errRawID[gIndex] = rID;
        continue;
      }
      /*rocp = converter.toRoc(link,roc); //Apparently this check is not needed, we don't have this conversion
      if unlikely(!rocp) {
          errorsInEvent = true;
          conversionError(fedId, &converter, 2, ww, errors);
          skipROC=true;
          continue;
      }*/
        
      DetIdGPU detId = getRawId(Map, fedId, link, roc);
      uint rawId  = detId.RawId;
      uint rocIdInDetUnit = detId.rocInDet;
     
      bool barrel = isBarrel(rawId);
        
      uint index = fedId * MAX_LINK * MAX_ROC + (link-1) * MAX_ROC + roc;
      if (useQualityInfo) {
          
          skipROC = Map->badRocs[index];
          if (skipROC) continue;
          
      }
      skipROC = Map->modToUnp[index];
      if (skipROC) continue;
  
      uint layer = 0;//, ladder =0;
      int side = 0, panel = 0, module = 0;//disk = 0,blade = 0
    
      if(barrel) {
        layer  = (rawId >> layerStartBit_) & layerMask_;
        //ladder = (rawId >> ladderStartBit_) & ladderMask_;
        module = (rawId >> moduleStartBit_) & moduleMask_;
        side   = (module<5)? -1 : 1;
      }
      else {
        // endcap ids
        layer = 0;
        panel = (rawId >> panelStartBit_) & panelMask_;
        //disk  = (rawId >> diskStartBit_)  & diskMask_ ;
        side  = (panel==1)? -1 : 1;
        //blade = (rawId>>bladeStartBit_) & bladeMask_;
      }
        
      // ***special case of layer to 1 be handled here
      Pixel localPix;
      if(layer == 1) {
        uint col = (ww >> COL_shift) & COL_mask;
        uint row = (ww >> ROW_shift) & ROW_mask;
        localPix.row = row;
        localPix.col = col;
          
        if( !rocRowColIsValid(row, col) && includeErrors) {
          uint error = conversionError(fedId, 3, debug); //use the device function and fill the arrays
          errType[gIndex] = error;
          errWord[gIndex] = ww;
          errFedID[gIndex] = fedId;
          errRawID[gIndex] = rawId;
          printf("Error status: %i\n", error);
          continue;
        }
          
      }
      else {
        // ***conversion rules for dcol and pxid
        uint dcol = (ww >> DCOL_shift) & DCOL_mask;
        uint pxid = (ww >> PXID_shift) & PXID_mask;
        uint row  = numRowsInRoc - pxid/2;
        uint col  = dcol*2 + pxid%2;
        localPix.row = row;
        localPix.col = col;
          
        if( !dcolIsValid(dcol, pxid) && includeErrors) {
          uint error = conversionError(fedId, 3, debug);
          errType[gIndex] = error;
          errWord[gIndex] = ww;
          errFedID[gIndex] = fedId;
          errRawID[gIndex] = rawId;
          printf("Error status: %i\n", error);
          continue;
	    }
        
      }
      
      Pixel globalPix = frameConversion(barrel, side, layer, rocIdInDetUnit, localPix);
      //printf("GPU side: %i, layer: %i, roc: %i, lrow: %i, lcol: %i, grow: %i, gcol: %i, word: %i\n", side, layer, rocIdInDetUnit, localPix.row, localPix.col, globalPix.row, globalPix.col, ww);
      XX[gIndex]    = globalPix.row  ; // origin shifting by 1 0-159
      YY[gIndex]    = globalPix.col ; // origin shifting by 1 0-415
      ADC[gIndex]   = getADC(ww);
      layerArr[gIndex] = layer;
      moduleId[gIndex] = detId.moduleId;
      rawIdArr[gIndex] = rawId;
        
      errType[gIndex] = 0;
      errWord[gIndex] = ww;
      errFedID[gIndex] = fedId;
      errRawID[gIndex] = rawId;
      //fill error type
    } // end of if(gIndex < end)
  } // end of for(int i =0;i<no_itr...)
  
  __syncthreads();

  // three cases possible
  // case 1: 21 21 21 22 21 22 22
  // pos   : 0  1  2  3  4  5  6
  // solution swap 21 with 22 : 21 21 21 21 22 22 22
  // atomicExch(address, value), set the variable at address to value.
  // do the swapping for above case and replace the 9999 with 
  // valid moduleId
   
  for(int i = 0; i < no_itr; i++) {
    uint gIndex = begin + threadId + i*blockDim.x;  
    if(gIndex <end) {
      //rare condition 
      if(moduleId[gIndex]==moduleId[gIndex+2] && moduleId[gIndex]<moduleId[gIndex+1]) {
        atomicExch(&moduleId[gIndex+2], atomicExch(&moduleId[gIndex+1], moduleId[gIndex+2]));
        //*swap all the digi id
        atomicExch(&XX[gIndex+2], atomicExch(&XX[gIndex+1], XX[gIndex+2]));
        atomicExch(&YY[gIndex+2], atomicExch(&YY[gIndex+1], YY[gIndex+2]));
        atomicExch(&ADC[gIndex+2], atomicExch(&ADC[gIndex+1], ADC[gIndex+2]));
        atomicExch(&layerArr[gIndex+2], atomicExch(&layerArr[gIndex+1], layerArr[gIndex+2]));
        atomicExch(&rawIdArr[gIndex+2], atomicExch(&rawIdArr[gIndex+1], rawIdArr[gIndex+2]));
      }
      __syncthreads();

      //rarest condition
      // above condition fails at 361 361 361 363 362 363 363
      // here we need to swap 362 with previous 363
      if(moduleId[gIndex]==moduleId[gIndex+2] && moduleId[gIndex]>moduleId[gIndex+1]) {
        atomicExch(&moduleId[gIndex+1], atomicExch(&moduleId[gIndex], moduleId[gIndex+1]));
        //*swap all the digi id
        atomicExch(&XX[gIndex+1], atomicExch(&XX[gIndex], XX[gIndex+1]));
        atomicExch(&YY[gIndex+1], atomicExch(&YY[gIndex], YY[gIndex+1]));
        atomicExch(&ADC[gIndex+1], atomicExch(&ADC[gIndex], ADC[gIndex+1]));
        atomicExch(&layerArr[gIndex+1], atomicExch(&layerArr[gIndex], layerArr[gIndex+1]));
        atomicExch(&rawIdArr[gIndex+1], atomicExch(&rawIdArr[gIndex], rawIdArr[gIndex+1]));
      }

      // moduleId== 9999 then pixel is bad with x=y=layer=adc=0
      // this bad pixel will not affect the cluster, since for cluster
      // the origin is shifted at (1,1) so x=y=0 will be ignored
      // assign the previous valid moduleId to this pixel to remove 9999
      // so that we can get the start & end index of module easily.
      __syncthreads(); // let the swapping finish first
        
      if(moduleId[gIndex] == 9999) {
        int m=gIndex;
        while(moduleId[--m] == 9999) {} //skip till you get the valid module
        moduleId[gIndex] = moduleId[m];
      } 
    } // end of if(gIndex<end)
  } //  end of for(int i=0;i<no_itr;...)
  __syncthreads();

  // mIndexStart stores starting index of module
  // mIndexEnd stores end index of module 
  // both indexes are inclusive 
  // check consecutive module numbers
  // for start of fed
  for(int i = 0; i < no_itr; i++) {
    uint gIndex = begin + threadId + i*blockDim.x; 
    uint moduleOffset = NMODULE*eventno; 
    //if(threadId==0) printf("moduleOffset: %u\n",moduleOffset );
    if(gIndex < end) {
      if(gIndex == begin) {
        mIndexStart[moduleOffset+moduleId[gIndex]] = gIndex;
      }
      // for end of the fed
      if(gIndex == (end-1)) {  
        mIndexEnd[moduleOffset+moduleId[gIndex]] = gIndex;
      }   
      // point to the gIndex where two consecutive moduleId varies
      if(gIndex!= begin && (gIndex<(end-1)) && moduleId[gIndex]!=9999) {
        if(moduleId[gIndex]<moduleId[gIndex+1] ) {
          mIndexEnd[moduleOffset + moduleId[gIndex]] = gIndex;
        }
        if(moduleId[gIndex] > moduleId[gIndex-1] ) {
          mIndexStart[moduleOffset+ moduleId[gIndex]] = gIndex;
        } 
      } //end of if(gIndex!= begin && (gIndex<(end-1)) ...  
    } //end of if(gIndex <end) 
  }
} // end of Raw to Digi kernel


// kernel wrapper called from runRawToDigi_kernel
void RawToDigi_wrapper (const CablingMap* cablingMapDevice, const uint wordCounter, uint *word, const uint fedCounter,  uint *fedIndex,
                        uint *eventIndex, bool convertADCtoElectrons, uint *xx_h, uint *yy_h, uint *adc_h, int *mIndexStart_h,
                        int *mIndexEnd_h, uint *rawIdArr_h, uint *errType_h, uint *errWord_h, uint *errFedID_h, uint *errRawID_h,
                        bool useQualityInfo, bool includeErrors, bool debug) {
  
 
  cout<<"Inside GPU RawToDigi , Total pixels: "<<wordCounter<<endl;

  const int threads = 512;
  const int blockX = 108; // only 108 feds are present
  const int blockY = NEVENT/NSTREAM;   //blockIdx.y is the no of events processed in kernel concurrently
  dim3 gridsize(blockX, blockY); 
  
  int MSIZE = NMODULE*NEVENT*sizeof(int)+sizeof(int);
  // initialize moduleStart & moduleEnd with some constant(-1)
  // just to check if it updated in kernel or not
  cudaMemset(mIndexStart_d, -1, MSIZE);
  cudaMemset(mIndexEnd_d, -1, MSIZE);
  cudaMemcpy(eventIndex_d, eventIndex, (NEVENT+1)*sizeof(uint), cudaMemcpyHostToDevice);
  
  int FSIZE = (blockY*2*MAX_FED +1)*sizeof(uint); // 0 to 150:fedId, 150:300: fedIndex
  
  int fedOffset  = 0;
  int wordOffset = 0;
  int wordSize   = 0;
  for (int i = 0; i < NSTREAM; i++) {
    fedOffset  = blockY*2*MAX_FED*i;
    wordOffset = eventIndex[blockY*i];
    // total no of words in blockY event to be trasfered on device 
    wordSize   = (eventIndex[blockY*(i+1)] - eventIndex[blockY*i]); 

    cudaMemcpyAsync(&word_d[wordOffset], &word[wordOffset], wordSize*sizeof(uint), cudaMemcpyHostToDevice, stream[i]);

    cudaMemcpyAsync(&fedIndex_d[fedOffset], &fedIndex[fedOffset], FSIZE, cudaMemcpyHostToDevice, stream[i]); 
    // Launch rawToDigi kernel

    RawToDigi_kernel<<<gridsize, threads, 0, stream[i]>>>(cablingMapDevice, word_d, fedIndex_d,eventIndex_d,i, xx_d, yy_d, moduleId_d,
                                        mIndexStart_d, mIndexEnd_d, adc_d,layer_d, rawIdArr_d, errType_d, errWord_d, errFedID_d, errRawID_d, useQualityInfo, includeErrors, debug);
  }
  
  checkCUDAError("Error in RawToDigi_kernel");
  for (int i = 0; i < NSTREAM; i++) {
    cudaStreamSynchronize(stream[i]);
    checkCUDAError("Error in cuda stream cudaStreamSynchronize");
  }
  checkCUDAError("Error in RawToDigi_kernel");
  //cudaDeviceSynchronize();  
 
  // kernel to apply adc threashold on the channel
  //ADCThreshold adcThreshold;
  //uint numThreads = 512;
  //uint numBlocks = wordCounter/512 +1;
  //applyADCthreshold_kernel<<<numBlocks, numThreads>>>(xx_d, yy_d,layer_d,adc_d, wordCounter, adcThreshold, xx_adc, yy_adc);
  //cudaDeviceSynchronize();
  //checkCUDAError("Error in applying ADC threshold");
  cout << "Raw data is converted into digi for " << NEVENT << "  Events" << endl;

  // copy data to host variable
  // if you want to copy data after applying ADC threshold
  if(convertADCtoElectrons) {
    cudaMemcpy(xx_h, xx_adc, wordCounter*sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(yy_h, yy_adc, wordCounter*sizeof(uint), cudaMemcpyDeviceToHost);
  }
  else {
    cudaMemcpy(xx_h, xx_d, wordCounter*sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(yy_h, yy_d, wordCounter*sizeof(uint), cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(adc_h, adc_d, wordCounter*sizeof(uint), cudaMemcpyDeviceToHost);
  cudaMemcpy(rawIdArr_h, rawIdArr_d, wordCounter*sizeof(uint), cudaMemcpyDeviceToHost);

  cudaMemcpy(mIndexStart_h, mIndexStart_d, NEVENT*NMODULE*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(mIndexEnd_h, mIndexEnd_d, NEVENT*NMODULE*sizeof(int), cudaMemcpyDeviceToHost);
    
  if(includeErrors){
    cudaMemcpy(errType_h, errType_d, wordCounter*sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(errWord_h, errWord_d, wordCounter*sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(errFedID_h, errFedID_d, wordCounter*sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(errRawID_h, errRawID_d, wordCounter*sizeof(uint), cudaMemcpyDeviceToHost);
  }

  // End  of Raw2Digi and passing data for cluserisation
  // PixelCluster_Wrapper(xx_adc , yy_adc, adc_d,wordCounter, mIndexStart_d, mIndexEnd_d);
}
