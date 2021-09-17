#ifndef _SiStripActionExecutor_h_
#define _SiStripActionExecutor_h_

#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"
#include "DQM/SiStripMonitorClient/interface/SiStripQualityChecker.h"
#include "DQM/SiStripMonitorClient/interface/SiStripSummaryCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMapCreator.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <map>
#include <vector>
#include <string>
#include <TTree.h>

class MonitorUserInterface;
class SiStripFedCabling;
class SiStripDetCabling;

class SiStripActionExecutor {
public:
  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;

  SiStripActionExecutor(edm::ParameterSet const& ps);
  virtual ~SiStripActionExecutor();

  bool readConfiguration();
  bool readTkMapConfiguration(const SiStripDetCabling* detCabling,
                              const TkDetMap* tkDetMap,
                              const TrackerTopology* tTopo);

  void saveMEs(DQMStore& dqm_store, std::string fname);
  void createSummary(DQMStore& dqm_store);
  void createSummaryOffline(DQMStore& dqm_store);
  void createTkMap(const edm::ParameterSet& tkmapPset, DQMStore& dqm_store, const std::string& map_type);
  void createOfflineTkMap(const edm::ParameterSet& tkmapPset,
                          DQMStore& dqm_store,
                          std::string& map_type,
                          const SiStripQuality* stripQuality);
  void createTkInfoFile(std::vector<std::string> tkhmap_names,
                        TTree* tkinfo_tree,
                        DQMStore& dqm_store,
                        const GeometricDet* geomDet);

  void createStatus(DQMStore& dqm_store);
  void fillDummyStatus();
  void fillStatus(DQMStore& dqm_store,
                  const SiStripDetCabling* cabling,
                  const TkDetMap* tkDetMap,
                  const TrackerTopology* tTopo);
  void fillStatusAtLumi(DQMStore& dqm_store);

  void createDummyShiftReport();
  void createShiftReport(DQMStore& dqm_store);
  void printReportSummary(MonitorElement* me, std::ostringstream& str_val, std::string name);
  void printShiftHistoParameters(DQMStore& dqm_store,
                                 std::map<std::string, std::vector<std::string>> const& layout_map,
                                 std::ostringstream& str_val);
  void printFaultyModuleList(DQMStore& dqm_store, std::ostringstream& str_val);
  void createFaultyModuleMEs(DQMStore& dqm_store);

private:
  std::vector<std::string> tkMapMENames{};

  std::unique_ptr<SiStripSummaryCreator> summaryCreator_{nullptr};
  std::unique_ptr<SiStripTrackerMapCreator> tkMapCreator_{nullptr};
  std::unique_ptr<SiStripQualityChecker> qualityChecker_{nullptr};
  std::unique_ptr<SiStripConfigWriter> configWriter_{nullptr};

  edm::ParameterSet const pSet_;
};
#endif
