#include "FWCore/Catalog/interface/FileLocator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include <boost/filesystem.hpp>
#include <iostream>

int main() {
  try {
    edmplugin::PluginManager::configure(edmplugin::standard::config());

    std::string config =
        "import FWCore.ParameterSet.Config as cms\n"
        "process = cms.Process('edmFileUtil')\n"
        "process.InitRootHandlers = cms.Service('InitRootHandlers')\n"
        "process.SiteLocalConfigService = cms.Service('SiteLocalConfigService')\n";

    //create the services
    std::unique_ptr<edm::ParameterSet> params;
    edm::makeParameterSets(config, params);
    edm::ServiceToken tempToken(edm::ServiceRegistry::createServicesFromConfig(std::move(params)));

    //make the services available
    edm::ServiceRegistry::Operate operate(tempToken);

    {
      edm::FileLocator fl("", false);

      const char* lfn[] = {"/store/group/bha/bho",
                           "/bha/bho",
                           "bha",
                           "file:bha",
                           "file:/bha/bho",
                           "/castor/cern.ch/cms/bha/bho",
                           "rfio:/castor/cern.ch/cms/bha/bho",
                           "rfio:/bha/bho"};
      int nfile = 8;

      std::cout << "lfn2pfn" << std::endl;
      for (int i = 0; i < nfile; ++i)
        std::cout << lfn[i] << " -> " << fl.pfn(lfn[i]) << std::endl;

      std::cout << "pfn2lfn" << std::endl;
      for (int i = 0; i < nfile; ++i)
        std::cout << lfn[i] << " -> " << fl.lfn(lfn[i]) << std::endl;
    }

    {
      std::string CMSSW_BASE(std::getenv("CMSSW_BASE"));
      std::string CMSSW_RELEASE_BASE(std::getenv("CMSSW_RELEASE_BASE"));
      std::string file_name("/src/FWCore/Catalog/test/override_catalog.xml");
      std::string full_file_name = boost::filesystem::exists((CMSSW_BASE + file_name).c_str())
                                       ? CMSSW_BASE + file_name
                                       : CMSSW_RELEASE_BASE + file_name;

      edm::FileLocator fl(("trivialcatalog_file:" + full_file_name + "?protocol=override").c_str(), false);

      const char* lfn[] = {
          "/store/unmerged/relval/CMSSW_3_8_0_pre3/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V2-v1/0666/"
          "80EC0BCD-D279-DF11-B1DB-0030487C90EE.root",
          "/store/group/bha/bho",
          "/bha/bho",
          "bha",
          "file:bha",
          "file:/bha/bho",
          "/castor/cern.ch/cms/bha/bho",
          "rfio:/castor/cern.ch/cms/bha/bho",
          "rfio:/bha/bho"};
      int nfile = 9;

      std::cout << "lfn2pfn" << std::endl;
      for (int i = 0; i < nfile; ++i)
        std::cout << lfn[i] << " -> " << fl.pfn(lfn[i]) << std::endl;

      std::cout << "pfn2lfn" << std::endl;
      for (int i = 0; i < nfile; ++i)
        std::cout << lfn[i] << " -> " << fl.lfn(lfn[i]) << std::endl;
    }

  } catch (cms::Exception const& e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  catch (...) {
    std::cout << "got a problem..." << std::endl;
    return 1;
  }

  return 0;
}
