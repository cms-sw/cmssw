#ifndef EventFilter_SiPixelRawToDigi_SiPixelFedCablingMap_H
#define EventFilter_SiPixelRawToDigi_SiPixelFedCablingMap_H

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include <string>
#include <map>
class SiPixelFedCablingTree;


class SiPixelFedCablingMap : public SiPixelFedCabling {

public: 

  SiPixelFedCablingMap(const SiPixelFedCablingTree *cab);

  SiPixelFedCablingMap(const std::string & version="") : theVersion(version) {}

  virtual ~SiPixelFedCablingMap() {}

  SiPixelFedCablingTree * cablingTree() const; 

  virtual std::string version() const { return theVersion; }

  virtual const sipixelobjects::PixelROC* findItem(
      const sipixelobjects::CablingPathToDetUnit & path) const;

  virtual std::vector<sipixelobjects::CablingPathToDetUnit> pathToDetUnit(uint32_t rawDetId) const;

  std::vector<unsigned int> fedIds() const;

  struct Key { unsigned int fed, link, roc; bool operator < (const Key & other) const; };

private:
  std::string theVersion;
  typedef std::map<Key, sipixelobjects::PixelROC> Map;
  Map theMap; 
};

#endif
