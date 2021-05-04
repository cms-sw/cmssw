#ifndef SiPixelTools_SiPixelFrameReverter_H
#define SiPixelTools_SiPixelFrameReverter_H

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"
#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "CondFormats/SiPixelObjects/interface/ElectronicIndex.h"
#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"

#include <cstdint>
#include <map>
#include <vector>

class TrackerGeometry;

class SiPixelFrameReverter {
public:
  SiPixelFrameReverter(const SiPixelFedCabling* map);

  void buildStructure(const TrackerGeometry*);

  // Function to test if detId exists
  bool hasDetUnit(uint32_t detId) const { return (DetToFedMap.find(detId) != DetToFedMap.end()); }

  // Function to convert offline addressing to online
  int toCabling(sipixelobjects::ElectronicIndex& cabling, const sipixelobjects::DetectorIndex& detector) const;

  // Function to find FedId given detId
  int findFedId(uint32_t detId);

  // Function to find Fed link given detId and pixel row and col on plaquette
  // returns -1 if link can't be found
  short findLinkInFed(uint32_t detId, sipixelobjects::GlobalPixel global);

  // Function to find Roc number on a link given detId and pixel row and col on plaquette
  // returns -1 if Roc can't be found
  short findRocInLink(uint32_t detId, sipixelobjects::GlobalPixel global);

  // Function to find the Roc number within a plaquette given detId and pixel row and col on plaquette
  // returns -1 if Roc can't be found
  short findRocInDet(uint32_t detId, sipixelobjects::GlobalPixel global);

  // Function to find local pixel given detId and pixel row and col on plaquette
  sipixelobjects::LocalPixel findPixelInRoc(uint32_t detId, sipixelobjects::GlobalPixel global);

private:
  const SiPixelFedCabling* map_;

  std::map<uint32_t, std::vector<sipixelobjects::CablingPathToDetUnit> > DetToFedMap;
};
#endif
