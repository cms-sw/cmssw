#ifndef SiPixelTools_SiPixelFrameReverter_H
#define SiPixelTools_SiPixelFrameReverter_H

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "CondFormats/SiPixelObjects/interface/ElectronicIndex.h"
#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <boost/cstdint.hpp>

class SiPixelFedCablingMap;

class SiPixelFrameReverter {
public:

  SiPixelFrameReverter(const edm::EventSetup&, const SiPixelFedCabling* map);

  void buildStructure(edm::EventSetup const&);

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

  const SiPixelFedCabling * map_;

  std::map< uint32_t,std::vector<sipixelobjects::CablingPathToDetUnit> > DetToFedMap;

};
#endif


