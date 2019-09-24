#ifndef SiPixelObjects_SiPixelFrameConverter_H
#define SiPixelObjects_SiPixelFrameConverter_H

#include "CondFormats/SiPixelObjects/interface/ElectronicIndex.h"
#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include <cstdint>

class SiPixelFrameConverter {
public:
  typedef sipixelobjects::PixelFEDCabling PixelFEDCabling;

  //  using PixelFEDCabling = sipixelobjects::PixelFEDCabling;

  SiPixelFrameConverter(const SiPixelFedCabling* map, int fedId);

  bool hasDetUnit(uint32_t radId) const;

  sipixelobjects::PixelROC const* toRoc(int link, int roc) const;

  int toDetector(const sipixelobjects::ElectronicIndex& cabling, sipixelobjects::DetectorIndex& detector) const {
    using namespace sipixelobjects;
    auto roc = toRoc(cabling.link, cabling.roc);
    if (!roc)
      return 2;
    LocalPixel::DcolPxid local = {cabling.dcol, cabling.pxid};
    if (!local.valid())
      return 3;

    GlobalPixel global = roc->toGlobal(LocalPixel(local));
    detector.rawId = roc->rawId();
    detector.row = global.row;
    detector.col = global.col;

    return 0;
  }

  int toCabling(sipixelobjects::ElectronicIndex& cabling, const sipixelobjects::DetectorIndex& detector) const;

private:
  int theFedId;
  const SiPixelFedCabling* theMap;
  SiPixelFedCablingTree const* theTree;
  const PixelFEDCabling* theFed;
};
#endif
