#ifndef _SiPixelActionExecutor_h_
#define _SiPixelActionExecutor_h_

#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigWriter.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>

// For Tracker Map
enum funcType { EachBinContent, Entries, Mean, Sum, WeightedSum };
#define PI_12 0.261799
#define PI 3.141592
#define PI_2 1.570796

// Number of HalfCylinders in Endcap or number of Shells in Barrel, which is bigger
#define NLev1 4
// Number of Disks in Endcap or number of Layers in Barrel, which is bigger
#define NLev2 3
// Number of Blades in Endcap or number of Ladders in Barrel, which is bigger
#define NLev3 22
// Number of Modules - different for Endcap and Barrel, which is bigger
#define NLev4 7

#define NCyl 4
#define NDisk 2
#define NBlade 12
#define NModuleE 7

#define NShell 4
#define NLayer 3
#define NModuleB 4

#define NPoints 5

// End for Tracker Map

class SiPixelActionExecutor {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  SiPixelActionExecutor(bool offlineXMLfile, bool Tier0Flag);
  ~SiPixelActionExecutor();

  void createSummary(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, bool isUpgrade);
  void bookDeviations(DQMStore::IBooker &iBooker, bool isUpgrade);
  void bookEfficiency(DQMStore::IBooker &iBooker, bool isUpgrade);
  void createEfficiency(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, bool isUpgrade);
  void fillEfficiency(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, bool isbarrel, bool isUpgrade);
  void fillEfficiencySummary(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter);
  void bookOccupancyPlots(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, bool hiRes, bool isbarrel);
  void bookOccupancyPlots(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, bool hiRes);
  void createOccupancy(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter);
  void normaliseAvDigiOcc(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter);
  void normaliseAvDigiOccVsLumi(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, int lumisec);
  bool readConfiguration(int &tkmap_freq,
                         int &sum_barrel_freq,
                         int &sum_endcap_freq,
                         int &sum_grandbarrel_freq,
                         int &sum_grandendcap_freq,
                         int &message_limit,
                         int &source_type,
                         int &calib_type);
  bool readConfiguration(int &tkmap_freq, int &summary_freq);
  void readConfiguration();
  int getLadder(const std::string &dname);
  int getBlade(const std::string &dname);

private:
  MonitorElement *getSummaryME(DQMStore::IBooker &iBooker,
                               DQMStore::IGetter &iGetter,
                               std::string me_name,
                               bool isUpgrade);
  MonitorElement *getFEDSummaryME(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, std::string me_name);
  void GetBladeSubdirs(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, std::vector<std::string> &blade_subdirs);
  void fillSummary(DQMStore::IBooker &iBooker,
                   DQMStore::IGetter &iGetter,
                   std::string dir_name,
                   std::vector<std::string> &me_names,
                   bool isbarrel,
                   bool isUpgrade);
  void fillFEDErrorSummary(DQMStore::IBooker &iBooker,
                           DQMStore::IGetter &iGetter,
                           std::string dir_name,
                           std::vector<std::string> &me_names);
  void fillGrandBarrelSummaryHistos(DQMStore::IBooker &iBooker,
                                    DQMStore::IGetter &iGetter,
                                    std::vector<std::string> &me_names,
                                    bool isUpgrade);
  void fillGrandEndcapSummaryHistos(DQMStore::IBooker &iBooker,
                                    DQMStore::IGetter &iGetter,
                                    std::vector<std::string> &me_names,
                                    bool isUpgrade);
  void getGrandSummaryME(DQMStore::IBooker &iBooker,
                         DQMStore::IGetter &iGetter,
                         int nbin,
                         std::string &me_name,
                         std::vector<MonitorElement *> &mes);

  void fillOccupancy(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, bool isbarrel);

  SiPixelConfigParser *configParser_;
  SiPixelConfigWriter *configWriter_;

  std::vector<std::string> summaryMENames;
  std::vector<std::string> tkMapMENames;

  int message_limit_;
  int source_type_;
  int calib_type_;
  int ndet_;
  bool offlineXMLfile_;
  bool Tier0Flag_;

  MonitorElement *OccupancyMap;
  MonitorElement *PixelOccupancyMap;
  MonitorElement *HitEfficiency_L1;
  MonitorElement *HitEfficiency_L2;
  MonitorElement *HitEfficiency_L3;
  MonitorElement *HitEfficiency_L4;
  MonitorElement *HitEfficiency_Dp1;
  MonitorElement *HitEfficiency_Dp2;
  MonitorElement *HitEfficiency_Dp3;
  MonitorElement *HitEfficiency_Dm1;
  MonitorElement *HitEfficiency_Dm2;
  MonitorElement *HitEfficiency_Dm3;
  MonitorElement *HitEfficiencySummary;
  MonitorElement *DEV_adc_Barrel;
  MonitorElement *DEV_ndigis_Barrel;
  MonitorElement *DEV_charge_Barrel;
  MonitorElement *DEV_nclusters_Barrel;
  MonitorElement *DEV_size_Barrel;
  MonitorElement *DEV_adc_Endcap;
  MonitorElement *DEV_ndigis_Endcap;
  MonitorElement *DEV_charge_Endcap;
  MonitorElement *DEV_nclusters_Endcap;
  MonitorElement *DEV_size_Endcap;

  TH2F *temp_H;
  TH2F *temp_1x2;
  TH2F *temp_1x5;
  TH2F *temp_2x3;
  TH2F *temp_2x4;
  TH2F *temp_2x5;
};
#endif
