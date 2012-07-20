#ifndef SherpackFetcher_h
#define SherpackFetcher_h

#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <stdint.h>
#include <fcntl.h>
#include "frontier_client/frontier-cpp.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace spf {

class SherpackFetcher {
public:
  SherpackFetcher(edm::ParameterSet const&);
  ~SherpackFetcher();
  int FnFileGet(std::string);
  const char *classname() const { return "SherpackFetcher"; }
  
private:


  std::string SherpaProcess;
  std::string SherpackLocation;
  std::string SherpackChecksum;
  
};

}

#endif
