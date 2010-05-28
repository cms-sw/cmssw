#ifndef _SiPixelDataQuality_h_
#define _SiPixelDataQuality_h_

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigWriter.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelLayoutParser.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "xgi/Utils.h"
#include "xgi/Method.h"

#include "TCanvas.h"
#include "TPaveText.h"
#include "TF1.h"
#include "TH2F.h"
#include "TGaxis.h"

#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include <map>
#include <boost/cstdint.hpp>

class DQMStore;
class SiPixelEDAClient;
class SiPixelWebInterface;
class SiPixelHistoPlotter;
class SiPixelDataQuality {

 public:

  SiPixelDataQuality(  bool                                      offlineXMLfile);
 ~SiPixelDataQuality();

  int getDetId(                 MonitorElement                          * mE) ;				

  void bookGlobalQualityFlag    (DQMStore                               * bei,
				 bool                                     Tier0Flag,
				 int                                      nFEDs);

  void computeGlobalQualityFlag (DQMStore                               * bei,
                                 bool                                     init,
				 int                                      nFEDs,
				 bool                                     Tier0Flag);
  
  void computeGlobalQualityFlagByLumi (DQMStore                         * bei,
                                       bool                               init,
				       int                                nFEDs,
				       bool                               Tier0Flag,
				       int                                nEvents_lastLS_);
  
  void fillGlobalQualityPlot    (DQMStore                               * bei,
                                 bool                                     init,
                                 edm::EventSetup const                  & eSetup,
				 int                                      nFEDs,
				 bool                                     Tier0Flag);
  
 private:

  bool  offlineXMLfile_;
  
  
  TH2F * allmodsMap;
  TH2F * errmodsMap;
  TH2F * goodmodsMap;
  int count;
  int errcount;
  bool gotDigis;
  
  int objectCount_;
  bool DONE_;
  
  
  ofstream myfile_;  
  int nevents_;
  bool endOfModules_;
  edm::ESHandle<SiPixelFedCablingMap> theCablingMap;
  
  // Final combined Data Quality Flags:
  MonitorElement * SummaryReportMap;
  MonitorElement * SummaryPixel;
  MonitorElement * SummaryBarrel;
  MonitorElement * SummaryEndcap;
  float qflag_;
  int allMods_, errorMods_, barrelMods_, endcapMods_;
/*  int errorModsL1_, barrelModsL1_;
  int errorModsL2_, barrelModsL2_;
  int errorModsL3_, barrelModsL3_;
  int errorModsDP1_, endcapModsDP1_;
  int errorModsDP2_, endcapModsDP2_;
  int errorModsDM1_, endcapModsDM1_;
  int errorModsDM2_, endcapModsDM2_; */
 
  // FEDErrors Cuts:
  MonitorElement * NErrorsBarrel;
  MonitorElement * NErrorsEndcap;
  MonitorElement * NErrorsFEDs;
  int n_errors_barrel_, n_errors_endcap_, n_errors_feds_;
  //int n_errors_barrelL1_, n_errors_barrelL2_, n_errors_barrelL3_;
  //int n_errors_endcapDP1_, n_errors_endcapDP2_, n_errors_endcapDM1_, n_errors_endcapDM2_;
  float barrel_error_flag_, endcap_error_flag_, feds_error_flag_;
  //float BarrelL1_error_flag_, BarrelL2_error_flag_, BarrelL3_error_flag_;
  //float EndcapDP1_error_flag_, EndcapDP2_error_flag_, EndcapDM1_error_flag_, EndcapDM2_error_flag_;
  //float BarrelL1_cuts_flag_[15], BarrelL2_cuts_flag_[15], BarrelL3_cuts_flag_[15];
  //float EndcapDP1_cuts_flag_[15], EndcapDP2_cuts_flag_[15], EndcapDM1_cuts_flag_[15], EndcapDM2_cuts_flag_[15];
  
  bool digiStatsBarrel, clusterStatsBarrel, trackStatsBarrel;
  int digiCounterBarrel, clusterCounterBarrel, trackCounterBarrel;
  bool digiStatsEndcap, clusterStatsEndcap, trackStatsEndcap;
  int digiCounterEndcap, clusterCounterEndcap, trackCounterEndcap;
/*  bool digiStatsBarrelL1, clusterStatsBarrelL1, trackStatsBarrelL1;
  int digiCounterBarrelL1, clusterCounterBarrelL1, trackCounterBarrelL1;
  bool digiStatsBarrelL2, clusterStatsBarrelL2, trackStatsBarrelL2;
  int digiCounterBarrelL2, clusterCounterBarrelL2, trackCounterBarrelL2;
  bool digiStatsBarrelL3, clusterStatsBarrelL3, trackStatsBarrelL3;
  int digiCounterBarrelL3, clusterCounterBarrelL3, trackCounterBarrelL3;
  bool digiStatsEndcapDP1, clusterStatsEndcapDP1, trackStatsEndcapDP1;
  int digiCounterEndcapDP1, clusterCounterEndcapDP1, trackCounterEndcapDP1;
  bool digiStatsEndcapDP2, clusterStatsEndcapDP2, trackStatsEndcapDP2;
  int digiCounterEndcapDP2, clusterCounterEndcapDP2, trackCounterEndcapDP2;
  bool digiStatsEndcapDM1, clusterStatsEndcapDM1, trackStatsEndcapDM1;
  int digiCounterEndcapDM1, clusterCounterEndcapDM1, trackCounterEndcapDM1;
  bool digiStatsEndcapDM2, clusterStatsEndcapDM2, trackStatsEndcapDM2;
  int digiCounterEndcapDM2, clusterCounterEndcapDM2, trackCounterEndcapDM2; */

  // Digis Cuts:
  MonitorElement * NDigisBarrel;
  MonitorElement * NDigisEndcap;
  MonitorElement * DigiChargeBarrel;
  MonitorElement * DigiChargeEndcap;
  
  // Cluster Cuts:
  MonitorElement * ClusterSizeBarrel;
  MonitorElement * ClusterSizeEndcap;
  MonitorElement * ClusterChargeBarrel;
  MonitorElement * ClusterChargeEndcap;
  MonitorElement * NClustersBarrel;
  MonitorElement * NClustersEndcap;
  
  // Track Cuts:
  MonitorElement * NPixelTracks;
};
#endif
