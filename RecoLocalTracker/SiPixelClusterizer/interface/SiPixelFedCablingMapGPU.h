#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPU_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPU_h

namespace pixelgpudetails {
  // Maximum fed for phase1 is 150 but not all of them are filled
  // Update the number FED based on maximum fed found in the cabling map
  constexpr unsigned int MAX_FED = 150;
  constexpr unsigned int MAX_LINK = 48;  // maximum links/channels for Phase 1
  constexpr unsigned int MAX_ROC = 8;
  constexpr unsigned int MAX_SIZE = MAX_FED * MAX_LINK * MAX_ROC;
  constexpr unsigned int MAX_SIZE_BYTE_BOOL = MAX_SIZE * sizeof(unsigned char);
}  // namespace pixelgpudetails

// TODO: since this has more information than just cabling map, maybe we should invent a better name?
struct SiPixelFedCablingMapGPU {
  alignas(128) unsigned int fed[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned int link[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned int roc[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned int RawId[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned int rocInDet[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned int moduleId[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned char badRocs[pixelgpudetails::MAX_SIZE];
  alignas(128) unsigned int size = 0;
};

#endif
