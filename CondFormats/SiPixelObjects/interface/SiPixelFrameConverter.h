#ifndef SiPixelObjects_SiPixelFrameConverter_H
#define SiPixelObjects_SiPixelFrameConverter_H


class SiPixelFedCablingMap;
namespace sipixelobjects { class PixelFEDCabling; }

#include <boost/cstdint.hpp>

class SiPixelFrameConverter {
public:

  SiPixelFrameConverter(const SiPixelFedCablingMap * map, int fedId); 

  struct CablingIndex { int link; int roc; int dcol; int pxid; };

  struct DetectorIndex { uint32_t rawId; int row; int col; };

  bool hasDetUnit(uint32_t radId) const;

  int toDetector(const CablingIndex & cabling, DetectorIndex & detector) const;

  int toCabling(CablingIndex & cabling, const DetectorIndex & detector) const;

private:

  const sipixelobjects::PixelFEDCabling & theFed; 
  
};
#endif
