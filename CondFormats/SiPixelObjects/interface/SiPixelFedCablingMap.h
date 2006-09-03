#ifndef SiPixelFedCablingMap_H
#define SiPixelFedCablingMap_H

#include <vector>
#include <map>
#include <string>

#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"

class SiPixelFedCablingMap {
public:
  typedef sipixelobjects::PixelFEDCabling PixelFEDCabling;

  SiPixelFedCablingMap(const std::string & version="") : theVersion(version) {}

  /// add cabling for one fed
  void addFed(const PixelFEDCabling& f);

  /// get fed identified by its id
  const PixelFEDCabling * fed(unsigned int idFed) const;

  std::vector<const PixelFEDCabling *> fedList() const;

  ///map version
  const std::string & version() const { return theVersion; }

  std::string print(int depth = 0) const;

private:
  std::string theVersion; 
  std::map<int, PixelFEDCabling> theFedCablings;
};
#endif
