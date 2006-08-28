#ifndef PixelToFEDAssociate_H
#define PixelToFEDAssociate_H

/** \class PixelToFEDAssociate
 *  Check to which FED pixel module belongs to.
 *  The associacions are read from the datafile
 */

#include <string>

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"


class PixelToFEDAssociate {
public:
  virtual ~PixelToFEDAssociate() {}

  /// version
  virtual std::string version() const = 0;

  /// FED id for module
  virtual int operator()(const PixelModuleName &) const = 0;
};
#endif 
