#ifndef SiStripMonitorSummary_SiStripBaseCondObjDQM_h
#define SiStripMonitorSummary_SiStripBaseCondObjDQM_h

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"  /// ADDITON OF TK_HISTO_MAP

#include "CommonTools/UtilAlgos/interface/DetIdSelector.h"

#include <map>
#include <sstream>
#include <string>
#include <vector>

class TrackerTopology;
class SiStripBaseCondObjDQM {
public:
  SiStripBaseCondObjDQM(const edm::EventSetup &eSetup,
                        edm::RunNumber_t iRun,
                        edm::ParameterSet const &hPSet,
                        edm::ParameterSet const &fPSet);

  virtual ~SiStripBaseCondObjDQM(){};

  virtual void getActiveDetIds(const edm::EventSetup &eSetup) = 0;

  void analysis(const edm::EventSetup &eSetup_);
  void analysisOnDemand(const edm::EventSetup &eSetup_, uint32_t detIdOnDemand);
  void analysisOnDemand(const edm::EventSetup &eSetup_, const std::vector<uint32_t> &detIdsOnDemand);
  void analysisOnDemand(const edm::EventSetup &eSetup_,
                        std::string requestedSubDetector,
                        uint32_t requestedSide,
                        uint32_t requestedLayer);

  std::vector<uint32_t> getCabledModules();
  void selectModules(std::vector<uint32_t> &detIds_, const TrackerTopology *tTopo);

  //    virtual void fillTopSummaryMEs()=0;

  virtual unsigned long long getCache(const edm::EventSetup &eSetup_) = 0;
  virtual void getConditionObject(const edm::EventSetup &eSetup_) = 0;

  virtual void end();

protected:
  struct ModMEs {
    ModMEs()
        : ProfileDistr(nullptr),
          CumulDistr(nullptr),
          SummaryOfProfileDistr(nullptr),
          SummaryOfCumulDistr(nullptr),
          SummaryDistr(nullptr) {
      ;
    }
    MonitorElement *ProfileDistr;
    MonitorElement *CumulDistr;
    MonitorElement *SummaryOfProfileDistr;
    MonitorElement *SummaryOfCumulDistr;
    MonitorElement *SummaryDistr;
  };

  void getModMEs(ModMEs &CondObj_ME, const uint32_t &detId_, const TrackerTopology *tTopo);
  void getSummaryMEs(ModMEs &CondObj_ME, const uint32_t &detId_, const TrackerTopology *tTopo);
  std::pair<std::string, uint32_t> getLayerNameAndId(const uint32_t &detId_, const TrackerTopology *tTopo);
  std::pair<std::string, uint32_t> getStringNameAndId(const uint32_t &detId_, const TrackerTopology *tTopo);
  std::vector<uint32_t> GetSameLayerDetId(const std::vector<uint32_t> &activeDetIds,
                                          uint32_t selDetId,
                                          const TrackerTopology *tTopo);

  virtual void fillModMEs(const std::vector<uint32_t> &selectedDetIds, const edm::EventSetup &es);
  virtual void fillSummaryMEs(const std::vector<uint32_t> &selectedDetIds, const edm::EventSetup &es);
  virtual void fillMEsForDet(const ModMEs &selModME_, uint32_t selDetId_, const TrackerTopology *tTopo) = 0;
  virtual void fillMEsForLayer(
      /*std::map<uint32_t, ModMEs> selModMEsMap_, */ uint32_t selDetId_, const TrackerTopology *tTopo) = 0;

  void fillTkMap(const uint32_t &detid, const float &value);

  SiStripDetInfoFileReader *reader;

  const edm::EventSetup &eSetup_;
  edm::ParameterSet hPSet_;
  edm::ParameterSet fPSet_;

  bool Mod_On_;
  bool HistoMaps_On_;
  bool SummaryOnLayerLevel_On_;
  bool SummaryOnStringLevel_On_;
  bool GrandSummary_On_;
  double minValue, maxValue;
  std::vector<int> tkMapScaler;

  // bool ActiveDetIds_On_;

  std::string CondObj_fillId_;
  std::string CondObj_name_;

  std::map<uint32_t, ModMEs> ModMEsMap_;
  std::map<uint32_t, ModMEs> SummaryMEsMap_;
  std::vector<uint32_t> activeDetIds;
  std::vector<uint32_t> all_DetIds;

  unsigned long long cacheID_memory;
  unsigned long long cacheID_current;

  std::unique_ptr<TkHistoMap> Tk_HM_;
  std::unique_ptr<TkHistoMap> Tk_HM_H;
  std::unique_ptr<TkHistoMap> Tk_HM_L;
  TrackerMap *tkMap;

private:
  void bookProfileMEs(SiStripBaseCondObjDQM::ModMEs &CondObj_ME, const uint32_t &detId_, const TrackerTopology *tTopo);
  void bookCumulMEs(SiStripBaseCondObjDQM::ModMEs &CondObj_ME, const uint32_t &detId_, const TrackerTopology *tTopo);
  void bookSummaryProfileMEs(SiStripBaseCondObjDQM::ModMEs &CondObj_ME,
                             const uint32_t &detId_,
                             const TrackerTopology *tTopo);
  void bookSummaryCumulMEs(SiStripBaseCondObjDQM::ModMEs &CondObj_ME,
                           const uint32_t &detId_,
                           const TrackerTopology *tTopo);
  void bookSummaryMEs(SiStripBaseCondObjDQM::ModMEs &CondObj_ME, const uint32_t &detId_, const TrackerTopology *tTopo);

  void bookTkMap(const std::string &TkMapname);

  void saveTkMap(const std::string &TkMapname, double minValue, double maxValue);

  std::vector<uint32_t> ModulesToBeExcluded_;
  std::vector<uint32_t> ModulesToBeIncluded_;
  std::vector<std::string> SubDetectorsToBeExcluded_;

  edm::ESHandle<SiStripDetCabling> detCablingHandle_;

  std::string condDataMonitoringMode_;

  SiStripHistoId hidmanager;
  SiStripFolderOrganizer folder_organizer;
  DQMStore *dqmStore_;
  edm::RunNumber_t runNumber_;
};

#endif
