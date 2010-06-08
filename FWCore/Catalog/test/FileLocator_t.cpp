#include "FWCore/Catalog/interface/FileLocator.h"


#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/PluginManager.h"


int main() {

  try {
    std::string config =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('edmFileUtil')\n"
      "process.InitRootHandlers = cms.Service('InitRootHandlers')\n"
      "process.SiteLocalConfigService = cms.Service('SiteLocalConfigService')\n";

    //create the services
    edm::ServiceToken tempToken = edm::ServiceRegistry::createServicesFromConfig(config);

    //make the services available
    edm::ServiceRegistry::Operate operate(tempToken);

    FileLocator fl;

    const char * lfn[] = {
      "/bha/bho",
      "bha",
      "file:bha",
      "file:/bha/bho",
      "/castor/cern.ch/cms/bha/bho",
      "rfio:/castor/cern.ch/cms/bha/bho",
      "rfio:/bha/bho"
    };
    int nfile=7;
    
    std::cout << "lfn2pfn" << std::endl;
    for (int i=0; i<7; ++i)
      std::cout << lfn[i] << " -> " fl.lfn2pfn(lfn[i]) << std::endl;
    
    std::cout << "pfn2lfn" << std::endl;
    for (int i=0; i<7; ++i)
      std::cout << lfn[i] << " -> " fl.pfn2lfn(lfn[i]) << std::endl;



  } catch (...) {
    std::cout << "got a problem..." << std::endl;
    return 1;
  }



  return 0;

}
