#ifndef EventFilter_SiPixelRawToDigi_SiPixelFedCablingMap_H
#define EventFilter_SiPixelRawToDigi_SiPixelFedCablingMap_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include <string>
#include <map>
#include<memory>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#define NO_DICT
#endif


class SiPixelFedCablingTree;


class SiPixelFedCablingMap : public SiPixelFedCabling {

public: 

  SiPixelFedCablingMap(const SiPixelFedCablingTree *cab);

  SiPixelFedCablingMap(const std::string & version="") : theVersion(version) {}

  void initializeRocs();

  ~SiPixelFedCablingMap() override {}

#ifdef NO_DICT
  std::unique_ptr<SiPixelFedCablingTree> cablingTree() const; 
#endif

  std::string version() const override { return theVersion; }

  const sipixelobjects::PixelROC* findItem(
      const sipixelobjects::CablingPathToDetUnit & path) const override;

  std::vector<sipixelobjects::CablingPathToDetUnit> pathToDetUnit(uint32_t rawDetId) const override;

  std::unordered_map<uint32_t, unsigned int> det2fedMap() const override;
  std::map< uint32_t,std::vector<sipixelobjects::CablingPathToDetUnit> > det2PathMap() const override;


  std::vector<unsigned int> fedIds() const;

  struct Key { unsigned int fed, link, roc; bool operator < (const Key & other) const; 
  COND_SERIALIZABLE;
};

private:
  std::string theVersion;
  typedef std::map<Key, sipixelobjects::PixelROC> Map;
  Map theMap; 

  COND_SERIALIZABLE;
};

#endif
