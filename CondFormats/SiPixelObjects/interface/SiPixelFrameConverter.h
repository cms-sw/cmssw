#ifndef SiPixelObjects_SiPixelFrameConverter_H
#define SiPixelObjects_SiPixelFrameConverter_H



#include "CondFormats/SiPixelObjects/interface/ElectronicIndex.h"
#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"

#include <boost/cstdint.hpp>

class SiPixelFrameConverter {
public:

  SiPixelFrameConverter(const SiPixelFedCabling* map, int fedId); 

  bool hasDetUnit(uint32_t radId) const;

  int toDetector(const sipixelobjects::ElectronicIndex & cabling, 
                       sipixelobjects::DetectorIndex & detector) const;

  int toCabling(       sipixelobjects::ElectronicIndex & cabling, 
                 const sipixelobjects::DetectorIndex & detector) const;

private:

  int theFedId;
  const SiPixelFedCabling* theMap;
  
};
#endif
