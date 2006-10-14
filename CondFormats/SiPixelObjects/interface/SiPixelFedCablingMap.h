#ifndef  SiPixelFedCablingMap_H
#define SiPixelFedCablingMap_H

#include <vector>

class PixelFEDCabling;

class SiPixelFedCablingMap {
public:
  SiPixelFedCablingMap();
  ~SiPixelFedCablingMap();

  /// add cabling for one fed
  void addFed(PixelFEDCabling* f);

  /// get defined  cabling for all feds
  std::vector<PixelFEDCabling *> cabling() const { return theFedCablings; }

private:
  std::vector<PixelFEDCabling *> theFedCablings;
};
#endif
