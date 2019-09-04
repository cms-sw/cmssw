#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/SiPixelPlugins/plugins/SiPixelLorentzAngle_PayloadInspector.cc"
#include "CondCore/SiPixelPlugins/plugins/SiPixelQuality_PayloadInspector.cc"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

int main(int argc, char** argv) {
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::vector<edm::ParameterSet> psets;
  edm::ParameterSet pSet;
  pSet.addParameter("@service_type", std::string("SiteLocalConfigService"));
  psets.push_back(pSet);
  edm::ServiceToken servToken(edm::ServiceRegistry::createSet(psets));
  edm::ServiceRegistry::Operate operate(servToken);

  std::string connectionString("frontier://FrontierProd/CMS_CONDITIONS");

  // Lorentz Angle

  std::string tag = "SiPixelLorentzAngle_v11_offline";
  std::string runTimeType = cond::time::timeTypeName(cond::runnumber);
  cond::Time_t start = boost::lexical_cast<unsigned long long>(303790);
  cond::Time_t end = boost::lexical_cast<unsigned long long>(324245);

  std::cout << "## Exercising Lorentz Angle plots " << std::endl;

  SiPixelLorentzAngleValues histo1;
  histo1.process(connectionString, tag, runTimeType, start, start);
  std::cout << histo1.data() << std::endl;

  SiPixelLorentzAngleValueComparisonSingleTag histo2;
  histo2.process(connectionString, tag, runTimeType, start, end);
  std::cout << histo2.data() << std::endl;

  SiPixelLorentzAngleByRegionComparisonSingleTag histo3;
  histo3.process(connectionString, tag, runTimeType, start, end);
  std::cout << histo3.data() << std::endl;

  SiPixelBPixLorentzAngleMap histo4;
  histo4.process(connectionString, tag, runTimeType, start, start);
  std::cout << histo4.data() << std::endl;

  SiPixelFPixLorentzAngleMap histo5;
  histo5.process(connectionString, tag, runTimeType, end, end);
  std::cout << histo5.data() << std::endl;

  // 2 tags comparisons

  std::string tag2 = "SiPixelLorentzAngle_2016_ultralegacymc_v2";
  cond::Time_t start2 = boost::lexical_cast<unsigned long long>(1);

  SiPixelLorentzAngleValueComparisonTwoTags histo6;
  histo6.processTwoTags(connectionString, tag, tag2, start, start2);
  std::cout << histo6.data() << std::endl;

  SiPixelLorentzAngleByRegionComparisonTwoTags histo7;
  histo7.processTwoTags(connectionString, tag, tag2, start, start2);
  std::cout << histo7.data() << std::endl;

  // SiPixelQuality

  tag = "SiPixelQuality_forDigitizer_phase1_2018_permanentlyBad";
  start = boost::lexical_cast<unsigned long long>(1);
  end = boost::lexical_cast<unsigned long long>(1);

  std::cout << "## Exercising SiPixelQuality plots " << std::endl;

  SiPixelBPixQualityMap histo8;
  histo8.process(connectionString, tag, runTimeType, start, start);
  std::cout << histo8.data() << std::endl;

  SiPixelFPixQualityMap histo9;
  histo9.process(connectionString, tag, runTimeType, start, start);
  std::cout << histo9.data() << std::endl;
}
