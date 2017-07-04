#include<iostream>
#include<sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/SiStripPlugins/plugins/SiStripApvGain_PayloadInspector.cc"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

int main(int argc, char** argv) {
  
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::vector<edm::ParameterSet> psets;
  edm::ParameterSet pSet;
  pSet.addParameter("@service_type",std::string("SiteLocalConfigService"));
  psets.push_back(pSet);
  edm::ServiceToken servToken(edm::ServiceRegistry::createSet(psets));
  edm::ServiceRegistry::Operate operate(servToken);

  std::string connectionString("frontier://FrontierProd/CMS_CONDITIONS");

  std::string tag = "SiStripApvGain_FromParticles_GR10_v10_offline";
  std::string runTimeType = cond::time::timeTypeName(cond::runnumber);
  cond::Time_t since = boost::lexical_cast<unsigned long long>(132440);

  std::cout <<"## TrackerMap Histo"<<std::endl;
  
  SiStripApvGainsAverageTrackerMap histo1;
  histo1.process( connectionString, tag, runTimeType, since, since );
  std::cout <<histo1.data()<<std::endl;

}
