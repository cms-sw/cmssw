#include<iostream>
#include<sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/SiStripPlugins/plugins/SiStripApvGain_PayloadInspector.cc"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

int main(int argc, char** argv) {
  
  if (argc < 2) {
    std::cout << "Not enough arguments given." << std::endl;
    return 0;
  }

  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::vector<edm::ParameterSet> psets;
  edm::ParameterSet pSet;
  pSet.addParameter("@service_type",std::string("SiteLocalConfigService"));
  psets.push_back(pSet);
  edm::ServiceToken servToken(edm::ServiceRegistry::createSet(psets));
  edm::ServiceRegistry::Operate operate(servToken);

  std::string connectionString("frontier://FrontierProd/CMS_CONDITIONS");

  std::string tag = "SiStripApvGain_FromParticles_GR10_v11_offline";
  std::string runTimeType = cond::time::timeTypeName(cond::runnumber);
  cond::Time_t start = boost::lexical_cast<unsigned long long>(132440);
  cond::Time_t end   = boost::lexical_cast<unsigned long long>(285368);

  std::cout <<"## TrackerMap Histo"<<std::endl;
  
  SiStripApvGainsAverageTrackerMap histo1;
  histo1.process( connectionString, tag, runTimeType, start, start );
  std::cout <<histo1.data()<<std::endl;

  SiStripApvGainsRatioWithPreviousIOVTrackerMap histo2;
  histo2.process( connectionString, tag, runTimeType, end, end );
  std::cout <<histo2.data()<<std::endl;

  SiStripApvGainsRatioMaxDeviationWithPreviousIOVTrackerMap histo3;
  histo3.process( connectionString, tag, runTimeType, end, end );
  std::cout <<histo3.data()<<std::endl;

}
