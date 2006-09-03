#ifndef SiPixelObjects_SiPixelFrameConverter_H
#define SiPixelObjects_SiPixelFrameConverter_H


class SiPixelFedCablingMap;
namespace sipixelobjects { class PixelROC; };
namespace sipixelobjects { class PixelFEDCabling; };

#include <boost/cstdint.hpp>

class SiPixelFrameConverter {
public:

  SiPixelFrameConverter(const SiPixelFedCablingMap * map, int fedId); 

  struct CablingIndex { int link; int roc; int dcol; int pxid; };
  struct DetectorIndex { uint32_t rawId; int row; int col; };


  bool hasDetUnit(uint32_t radId) const;

  DetectorIndex toDetector(const CablingIndex & cabling) const;

  CablingIndex toCabling(const DetectorIndex & detector) const;

private:
 

  /// local coordinates in this ROC (double column, pixelid in double column)
  struct LocalPixel { int dcol, pxid; };

  /// global coordinates (row and column in DetUnit, as in PixelDigi)
  struct GlobalPixel { int row; int col; };

  LocalPixel  toLocal(const sipixelobjects::PixelROC& roc, const GlobalPixel & duc) const;
  GlobalPixel toGlobal(const sipixelobjects::PixelROC& roc, const LocalPixel& loc) const;

private:

  const sipixelobjects::PixelFEDCabling & theFed; 
  
};
#endif
