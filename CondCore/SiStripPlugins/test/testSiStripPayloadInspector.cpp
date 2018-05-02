#include<iostream>
#include<sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/SiStripPlugins/plugins/SiStripApvGain_PayloadInspector.cc"
#include "CondCore/SiStripPlugins/plugins/SiStripNoises_PayloadInspector.cc"
#include "CondCore/SiStripPlugins/plugins/SiStripPedestals_PayloadInspector.cc"

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

  // Gains

  std::string tag = "SiStripApvGain_FromParticles_GR10_v11_offline";
  std::string runTimeType = cond::time::timeTypeName(cond::runnumber);
  cond::Time_t start = boost::lexical_cast<unsigned long long>(132440);
  cond::Time_t end   = boost::lexical_cast<unsigned long long>(285368);

  std::cout <<"## Exercising Gains plots "<<std::endl;
  
  SiStripApvGainsAverageTrackerMap histo1;
  histo1.process( connectionString, tag, runTimeType, start, start );
  std::cout <<histo1.data()<<std::endl;

  SiStripApvGainsAvgDeviationRatio1sigmaTrackerMap histo2;
  histo2.process( connectionString, tag, runTimeType, start, end );
  std::cout <<histo2.data()<<std::endl;

  SiStripApvGainsMaxDeviationRatio1sigmaTrackerMap histo3;
  histo3.process( connectionString, tag, runTimeType, start, end );
  std::cout <<histo3.data()<<std::endl;

  SiStripApvGainsValuesComparator histo4;
  histo4.process( connectionString, tag, runTimeType, start, end );
  std::cout <<histo4.data()<<std::endl;

  SiStripApvGainsComparator histo5;
  histo5.process( connectionString, tag, runTimeType, start, end );
  std::cout <<histo5.data()<<std::endl;

  SiStripApvGainsComparatorByRegion histo6;
  histo6.process( connectionString, tag, runTimeType, start, end );
  std::cout <<histo6.data()<<std::endl;

  SiStripApvGainsRatioComparatorByRegion histo7;
  histo7.process( connectionString, tag, runTimeType, start, end );
  std::cout <<histo7.data()<<std::endl;

  // Noise

  tag   = "SiStripNoise_GR10_v1_hlt";
  start = boost::lexical_cast<unsigned long long>(312968);
  end   = boost::lexical_cast<unsigned long long>(313120);

  std::cout <<"## Exercising Noise plots "<<std::endl;

  SiStripNoiseValuePerAPV histo8;
  histo8.process(connectionString, tag, runTimeType, start, start );
  std::cout <<histo8.data()<<std::endl;

  SiStripNoiseValueComparisonPerAPV histo9;
  histo9.process(connectionString, tag, runTimeType, start, end);
  std::cout <<histo9.data()<<std::endl;

  // Pedestals

  tag   = "SiStripPedestals_v2_prompt";
  start = boost::lexical_cast<unsigned long long>(303420);
  end   = boost::lexical_cast<unsigned long long>(313120);

  std::cout <<"## Exercising Pedestal plots "<<std::endl;

  SiStripPedestalValuePerStrip histo10;
  histo10.process(connectionString, tag, runTimeType, start, start );
  std::cout <<histo8.data()<<std::endl;

  SiStripPedestalValueComparisonPerModule histo11;
  histo11.process(connectionString, tag, runTimeType, start, end);
  std::cout <<histo9.data()<<std::endl;
}

