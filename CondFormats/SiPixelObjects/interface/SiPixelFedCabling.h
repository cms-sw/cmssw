#ifndef EventFilter_SiPixelRawToDigi_SiPixelFedCabling_H
#define EventFilter_SiPixelRawToDigi_SiPixelFedCabling_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"
#include <vector>

class SiPixelFedCabling {

public:

  virtual ~SiPixelFedCabling() {}

  virtual std::string version() const = 0;

  virtual const sipixelobjects::PixelROC* findItem(
      const sipixelobjects::CablingPathToDetUnit&) const = 0;

  virtual std::vector<sipixelobjects::CablingPathToDetUnit> pathToDetUnit(
      uint32_t rawDetId) const = 0;

  virtual std::unordered_map<uint32_t, unsigned int> det2fedMap() const =0; 

  virtual std::map< uint32_t,std::vector<sipixelobjects::CablingPathToDetUnit> > det2PathMap() const=0;


  COND_SERIALIZABLE;
};

#endif

