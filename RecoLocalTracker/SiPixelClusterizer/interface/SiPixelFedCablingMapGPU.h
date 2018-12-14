#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPU_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPU_h

namespace pixelgpudetails {
  // Maximum fed for phase1 is 150 but not all of them are filled
  // Update the number FED based on maximum fed found in the cabling map
  constexpr unsigned int MAX_FED  = 150;
  constexpr unsigned int MAX_LINK =  48;  // maximum links/channels for Phase 1
  constexpr unsigned int MAX_ROC  =   8;
  constexpr unsigned int MAX_SIZE = MAX_FED * MAX_LINK * MAX_ROC;
  constexpr unsigned int MAX_SIZE_BYTE_INT  = MAX_SIZE * sizeof(unsigned int);
  constexpr unsigned int MAX_SIZE_BYTE_BOOL = MAX_SIZE * sizeof(unsigned char);
}

// TODO: since this has more information than just cabling map, maybe we should invent a better name?
struct SiPixelFedCablingMapGPU {
  unsigned int size = 0;
  unsigned int * fed = nullptr;
  unsigned int * link = nullptr;
  unsigned int * roc = nullptr;
  unsigned int * RawId = nullptr;
  unsigned int * rocInDet = nullptr;
  unsigned int * moduleId = nullptr;
  unsigned char * badRocs = nullptr;
};

#endif
