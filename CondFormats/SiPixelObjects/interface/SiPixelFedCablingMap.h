#ifndef EventFilter_SiPixelRawToDigi_SiPixelFedCablingMap_H
#define EventFilter_SiPixelRawToDigi_SiPixelFedCablingMap_H

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include <string>
#include <map>

class SiPixelFedCablingMap {

public: 

  SiPixelFedCablingMap(const SiPixelFedCablingTree *cab);

  SiPixelFedCablingMap(const std::string & version="") : theVersion(version) {}

  virtual ~SiPixelFedCablingMap() {}

  SiPixelFedCablingTree * cablingTree() const; 

  std::string version() const { return theVersion; }

  const sipixelobjects::PixelROC* findItem(unsigned int fedId, unsigned int linkId, unsigned int rocId) const;

  struct Key { unsigned int fed, link, roc; bool operator < (const Key & other) const; };

private:
  std::string theVersion;
  typedef std::map<Key, sipixelobjects::PixelROC> Map;
  Map theMap; 
};

#endif
