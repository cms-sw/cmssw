#ifndef SiPixelObjects_SiPixelFrameConverter_H
#define SiPixelObjects_SiPixelFrameConverter_H


class SiPixelFedCablingMap;
namespace sipixelobjects { class PixelROC; };
namespace sipixelobjects { class PixelFEDCabling; };
#include "CondFormats/SiPixelObjects/interface/ModuleType.h"

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

  LocalPixel  toLocal( const sipixelobjects::PixelROC& roc, const GlobalPixel & duc) const;

  /// convert local coordinates in ROC (dcol,pxid) to Module-wide frame.
  //  Module frame is defined by axes of first ROC 
  GlobalPixel toGlobal( const sipixelobjects::PixelROC& roc, const LocalPixel& loc) const;

  /// number of rows in columns in a module of given type
  std::pair<int,int> rowsAndCollumns(const sipixelobjects::ModuleType & t) const;

  /// conversion between Module frame and DetUnit frame. Module frame is 
  /// defined by first pixel ROC. DetUnit - as in CMSSW
  void convertModuleToDetUnit(const sipixelobjects::ModuleType & t, 
       int nRowsModule, int nColsModule, GlobalPixel & coorinates) const;

  ///void reverse DetUnit frame for modules which are not correctly rotated in geom
  void reverseDetUnitFrame( const uint32_t & detId, 
      int nRowsModule, int nColsModule, GlobalPixel & global) const;

private:

  const sipixelobjects::PixelFEDCabling & theFed; 
  
};
#endif
