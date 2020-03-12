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
  struct CablingRocId {
    int fedId;
    int linkId;
    int rocLinkId;
  };
  struct DetectorRocId {
    const PixelModuleName *module;
    int rocDetId;
  };

  virtual ~PixelToFEDAssociate() {}

  /// version
  virtual std::string version() const = 0;

  /// FED id for module
  virtual int operator()(const PixelModuleName &) const { return 0; }

  /// LNK id for module
  virtual const CablingRocId *operator()(const DetectorRocId &roc) const { return nullptr; }
};
#endif
