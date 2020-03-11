#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/AlignmentPlugins/plugins/TrackerAlignment_PayloadInspector.cc"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
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

  std::string tag = "TrackerAlignment_v21_offline";
  std::string runTimeType = cond::time::timeTypeName(cond::runnumber);
  cond::Time_t start = boost::lexical_cast<unsigned long long>(294034);
  cond::Time_t end = boost::lexical_cast<unsigned long long>(305898);

  std::cout << "## Alignment Histos" << std::endl;

  TrackerAlignmentCompareX histo1;
  histo1.process(connectionString, tag, runTimeType, start, end);
  std::cout << histo1.data() << std::endl;

  TrackerAlignmentSummaryBPix histo2;
  histo2.process(connectionString, tag, runTimeType, start, end);
  std::cout << histo2.data() << std::endl;

  X_BPixBarycenterHistory histo3;
  histo3.process(connectionString, tag, runTimeType, start, end);
  std::cout << histo3.data() << std::endl;

  std::cout << "## Testing Two Tag Histos" << std::endl;

  TrackerAlignmentBarycentersCompareTwoTags histo4;
  histo4.processTwoTags(
      connectionString, "TrackerAlignment_2017_ultralegacymc_v2", "TrackerAlignment_Upgrade2017_realistic_v2", 1, 1);
  std::cout << histo4.data() << std::endl;

  TrackerAlignmentCompareXTwoTags histo5;
  histo5.processTwoTags(
      connectionString, "TrackerAlignment_2017_ultralegacymc_v2", "TrackerAlignment_Upgrade2017_realistic_v2", 1, 1);
  std::cout << histo5.data() << std::endl;

  std::cout << "## Testing Barycenter Histos" << std::endl;

  TrackerAlignmentBarycentersCompare histo6;
  histo6.process(connectionString, tag, runTimeType, start, end);
  std::cout << histo6.data() << std::endl;

  PixelBarycentersCompare histo7;
  histo7.process(connectionString, tag, runTimeType, start, end);
  std::cout << histo7.data() << std::endl;
}
