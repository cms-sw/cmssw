#include "FWCore/Catalog/interface/FileLocator.h"


#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/PluginManager.h"


int main() {

  try {
    edmplugin::PluginManager::configure(edmplugin::standard::config());

    std::string config =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('edmFileUtil')\n"
      "process.InitRootHandlers = cms.Service('InitRootHandlers')\n"
      "process.SiteLocalConfigService = cms.Service('SiteLocalConfigService')\n";

    //create the services
    edm::ServiceToken tempToken = edm::ServiceRegistry::createServicesFromConfig(config);

    //make the services available
    edm::ServiceRegistry::Operate operate(tempToken);

    {
      edm::FileLocator fl("");

      const char * lfn[] = {
      "/store/group/bha/bho",
      "/bha/bho",
      "bha",
      "file:bha",
      "file:/bha/bho",
      "/castor/cern.ch/cms/bha/bho",
      "rfio:/castor/cern.ch/cms/bha/bho",
      "rfio:/bha/bho"
      };
      int nfile=8;
    
      std::cout << "lfn2pfn" << std::endl;
      for (int i=0; i<nfile; ++i)
        std::cout << lfn[i] << " -> " << fl.pfn(lfn[i]) << std::endl;
    
      std::cout << "pfn2lfn" << std::endl;
      for (int i=0; i<nfile; ++i)
        std::cout << lfn[i] << " -> " << fl.lfn(lfn[i]) << std::endl;
   }

    {
      edm::FileLocator fl("trivialcatalog_file:FWCore/Catalog/test/override_catalog.xml?protocol=override");
    
      const char * lfn[] = {
      "/store/unmerged/relval/CMSSW_3_8_0_pre3/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V2-v1/0666/80EC0BCD-D279-DF11-B1DB-0030487C90EE.root",
      "/store/group/bha/bho",
      "/bha/bho",
      "bha",
      "file:bha",
      "file:/bha/bho",
      "/castor/cern.ch/cms/bha/bho",
      "rfio:/castor/cern.ch/cms/bha/bho",
      "rfio:/bha/bho"
      };
      int nfile=9;
    
      std::cout << "lfn2pfn" << std::endl;
      for (int i=0; i<nfile; ++i)
        std::cout << lfn[i] << " -> " << fl.pfn(lfn[i]) << std::endl;
    
      std::cout << "pfn2lfn" << std::endl;
      for (int i=0; i<nfile; ++i)
        std::cout << lfn[i] << " -> " << fl.lfn(lfn[i]) << std::endl;
   }


  } 
  catch (cms::Exception const & e) {
    std::cout << e.what()  << std::endl;
  }
  catch (...) {
    std::cout << "got a problem..." << std::endl;
    return 1;
  }



  return 0;

}
