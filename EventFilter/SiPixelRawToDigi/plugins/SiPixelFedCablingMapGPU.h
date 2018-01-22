#ifndef SiPixelFedCablingMapGPU_h
#define SiPixelFedCablingMapGPU_h

#include <set>

#include "cudaCheck.h"

class SiPixelFedCablingMap;
class SiPixelQuality;

// Maximum fed for phase1 is 150 but not all of them are filled
// Update the number FED based on maximum fed found in the cabling map
const unsigned int MAX_FED  = 150;
const unsigned int MAX_LINK =  48;  // maximum links/channels for Phase 1
const unsigned int MAX_ROC  =   8;
const unsigned int MAX_SIZE = MAX_FED * MAX_LINK * MAX_ROC;
const unsigned int MAX_SIZE_BYTE_INT  = MAX_SIZE * sizeof(unsigned int);
const unsigned int MAX_SIZE_BYTE_CHAR = MAX_SIZE * sizeof(unsigned char);

struct SiPixelFedCablingMapGPU {
  unsigned int size;
  unsigned int * fed;
  unsigned int * link;
  unsigned int * roc;
  unsigned int * RawId;
  unsigned int * rocInDet;
  unsigned int * moduleId;
  unsigned char * modToUnp;
  unsigned char * badRocs;
};

inline
void allocateCablingMap(SiPixelFedCablingMapGPU* & cablingMapHost, SiPixelFedCablingMapGPU* & cablingMapDevice) {
  cablingMapHost = new SiPixelFedCablingMapGPU();
  cudaCheck(cudaMalloc((void**) & cablingMapDevice, sizeof(SiPixelFedCablingMapGPU)));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->fed,      MAX_SIZE_BYTE_INT));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->link,     MAX_SIZE_BYTE_INT));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->roc,      MAX_SIZE_BYTE_INT));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->RawId,    MAX_SIZE_BYTE_INT));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->rocInDet, MAX_SIZE_BYTE_INT));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->moduleId, MAX_SIZE_BYTE_INT));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->badRocs,  MAX_SIZE_BYTE_CHAR));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->modToUnp, MAX_SIZE_BYTE_CHAR));
  cudaCheck(cudaMemcpy(cablingMapDevice, cablingMapHost, sizeof(SiPixelFedCablingMapGPU), cudaMemcpyHostToDevice));
}

inline
void deallocateCablingMap(SiPixelFedCablingMapGPU* cablingMapHost, SiPixelFedCablingMapGPU* cablingMapDevice) {
  cudaCheck(cudaFree(cablingMapHost->fed));
  cudaCheck(cudaFree(cablingMapHost->link));
  cudaCheck(cudaFree(cablingMapHost->roc));
  cudaCheck(cudaFree(cablingMapHost->RawId));
  cudaCheck(cudaFree(cablingMapHost->rocInDet));
  cudaCheck(cudaFree(cablingMapHost->moduleId));
  cudaCheck(cudaFree(cablingMapHost->modToUnp));
  cudaCheck(cudaFree(cablingMapHost->badRocs));
  cudaCheck(cudaFree(cablingMapDevice));
  delete cablingMapHost;
}

void processCablingMap(SiPixelFedCablingMap const& cablingMap, SiPixelFedCablingMapGPU* cablingMapGPU, SiPixelFedCablingMapGPU* cablingMapDevice, const SiPixelQuality* badPixelInfo, std::set<unsigned int> const& modules);

#endif // SiPixelFedCablingMapGPU_h
