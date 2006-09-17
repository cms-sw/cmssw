#ifndef SiPixelObjects_FrameConversion_H
#define SiPixelObjects_FrameConversion_H

#include "CondFormats/SiPixelObjects/interface/LinearConversion.h"
#include <boost/cstdint.hpp>

class PixelEndcapName;
class PixelBarrelName;

namespace sipixelobjects {

class FrameConversion {
public:
  FrameConversion( const PixelEndcapName & name, int rocIdInDetUnit);
  FrameConversion( const PixelBarrelName & name, int rocIdInDetUnit);
  FrameConversion( int rowOffset, int rowSlopeSign, int colOffset, int colSlopeSign);

  const sipixelobjects::LinearConversion & row() const { return theRowConversion; }
  const sipixelobjects::LinearConversion & collumn() const { return theCollumnConversion;}

private:
  sipixelobjects::LinearConversion theRowConversion;
  sipixelobjects::LinearConversion theCollumnConversion;
};

}
#endif
