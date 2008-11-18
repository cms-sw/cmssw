#ifndef SiPixelTools_SiPixelFrameReverter_H
#define SiPixelTools_SiPixelFrameReverter_H

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "CondFormats/SiPixelObjects/interface/ElectronicIndex.h"
#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <boost/cstdint.hpp>


using namespace sipixelobjects;

class SiPixelFedCablingMap;

class SiPixelFrameReverter {
public:

  SiPixelFrameReverter(const edm::EventSetup&, const SiPixelFedCablingTree *);

  typedef std::pair<int,SiPixelFrameConverter*> FEDType;

  void buildStructure(edm::EventSetup const&);

  int findFedId(uint32_t detId);

  int findLinkInFed(uint32_t, int, int);

  int findRocInLink(uint32_t, int, int);

  int findRocInDet(uint32_t, int, int);

  LocalPixel findPixelInRoc(uint32_t, int, int);

  //  LocalPixel::DcolPxid findPixelInDcol(uint32_t, int, int);

private:

  const SiPixelFedCablingTree * tree_;

  std::map<uint32_t,FEDType> DetToFedMap;

};
#endif
