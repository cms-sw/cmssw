#ifndef EventFilter_SiPixelRawToDigi_SiPixelFedCabling_H
#define EventFilter_SiPixelRawToDigi_SiPixelFedCabling_H

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"
#include <vector>

class SiPixelFedCabling {

public:

  virtual ~SiPixelFedCabling() {}

  virtual std::string version() const = 0;

  virtual const sipixelobjects::PixelROC* findItem(
      const sipixelobjects::CablingPathToDetUnit&) const = 0;

  virtual std::vector<sipixelobjects::CablingPathToDetUnit> pathToDetUnit(
      uint32_t rawDetId) const = 0;
};

#endif

