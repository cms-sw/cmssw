#ifndef SiPixelFedCablingMap_H
#define SiPixelFedCablingMap_H

#include <vector>
#include <string>

#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"

class SiPixelFedCablingMap {
public:
  SiPixelFedCablingMap(const std::string & version="") : theVersion(version) {}

  /// add cabling for one fed
  void addFed(const PixelFEDCabling& f);

  /// get defined  cabling for all feds
  const std::vector<PixelFEDCabling> & cabling() const { return theFedCablings; }

  /// get fed identified by its id
  const PixelFEDCabling * fed(unsigned int idFed) const;

  ///map version
  const std::string & version() const { return theVersion; }

  std::string print(int depth = 0) const;

private:
  std::string theVersion; 
  std::vector<PixelFEDCabling> theFedCablings;
};
#endif
