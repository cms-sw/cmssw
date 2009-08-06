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
  MonitorElement * SummaryReport;
  MonitorElement * SummaryReportMap;
  MonitorElement * SummaryPixel;
  MonitorElement * SummaryBarrel;
  MonitorElement * SummaryEndcap;
  float qflag_;
  int allMods_, errorMods_, barrelMods_, endcapMods_;
  int errorModsL1_, barrelModsL1_;
  int errorModsL2_, barrelModsL2_;
  int errorModsL3_, barrelModsL3_;
  int errorModsDP1_, endcapModsDP1_;
  int errorModsDP2_, endcapModsDP2_;
  int errorModsDM1_, endcapModsDM1_;
  int errorModsDM2_, endcapModsDM2_;
 
  // FEDErrors Cuts:
  MonitorElement * NErrorsBarrel;
  MonitorElement * NErrorsEndcap;
  MonitorElement * NErrorsFEDs;
  int n_errors_barrel_, n_errors_endcap_, n_errors_feds_;
  int n_errors_barrelL1_, n_errors_barrelL2_, n_errors_barrelL3_;
  int n_errors_endcapDP1_, n_errors_endcapDP2_, n_errors_endcapDM1_, n_errors_endcapDM2_;
  float barrel_error_flag_, endcap_error_flag_, feds_error_flag_;
  float BarrelL1_error_flag_, BarrelL2_error_flag_, BarrelL3_error_flag_;
  float EndcapDP1_error_flag_, EndcapDP2_error_flag_, EndcapDM1_error_flag_, EndcapDM2_error_flag_;
  float BarrelL1_cuts_flag_[14], BarrelL2_cuts_flag_[14], BarrelL3_cuts_flag_[14];
  float EndcapDP1_cuts_flag_[14], EndcapDP2_cuts_flag_[14], EndcapDM1_cuts_flag_[14], EndcapDM2_cuts_flag_[14];
  
  bool digiStatsBarrel, clusterOntrackStatsBarrel, clusterOfftrackStatsBarrel, rechitStatsBarrel, trackStatsBarrel;
  int digiCounterBarrel, clusterOntrackCounterBarrel, clusterOfftrackCounterBarrel, rechitCounterBarrel, trackCounterBarrel;
  bool digiStatsEndcap, clusterOntrackStatsEndcap, clusterOfftrackStatsEndcap, rechitStatsEndcap, trackStatsEndcap;
  int digiCounterEndcap, clusterOntrackCounterEndcap, clusterOfftrackCounterEndcap, rechitCounterEndcap, trackCounterEndcap;
  bool digiStatsBarrelL1, clusterOntrackStatsBarrelL1, clusterOfftrackStatsBarrelL1, rechitStatsBarrelL1, trackStatsBarrelL1;
  int digiCounterBarrelL1, clusterOntrackCounterBarrelL1, clusterOfftrackCounterBarrelL1, rechitCounterBarrelL1, trackCounterBarrelL1;
  bool digiStatsBarrelL2, clusterOntrackStatsBarrelL2, clusterOfftrackStatsBarrelL2, rechitStatsBarrelL2, trackStatsBarrelL2;
  int digiCounterBarrelL2, clusterOntrackCounterBarrelL2, clusterOfftrackCounterBarrelL2, rechitCounterBarrelL2, trackCounterBarrelL2;
  bool digiStatsBarrelL3, clusterOntrackStatsBarrelL3, clusterOfftrackStatsBarrelL3, rechitStatsBarrelL3, trackStatsBarrelL3;
  int digiCounterBarrelL3, clusterOntrackCounterBarrelL3, clusterOfftrackCounterBarrelL3, rechitCounterBarrelL3, trackCounterBarrelL3;
  bool digiStatsEndcapDP1, clusterOntrackStatsEndcapDP1, clusterOfftrackStatsEndcapDP1, rechitStatsEndcapDP1, trackStatsEndcapDP1;
  int digiCounterEndcapDP1, clusterOntrackCounterEndcapDP1, clusterOfftrackCounterEndcapDP1, rechitCounterEndcapDP1, trackCounterEndcapDP1;
  bool digiStatsEndcapDP2, clusterOntrackStatsEndcapDP2, clusterOfftrackStatsEndcapDP2, rechitStatsEndcapDP2, trackStatsEndcapDP2;
  int digiCounterEndcapDP2, clusterOntrackCounterEndcapDP2, clusterOfftrackCounterEndcapDP2, rechitCounterEndcapDP2, trackCounterEndcapDP2;
  bool digiStatsEndcapDM1, clusterOntrackStatsEndcapDM1, clusterOfftrackStatsEndcapDM1, rechitStatsEndcapDM1, trackStatsEndcapDM1;
  int digiCounterEndcapDM1, clusterOntrackCounterEndcapDM1, clusterOfftrackCounterEndcapDM1, rechitCounterEndcapDM1, trackCounterEndcapDM1;
  bool digiStatsEndcapDM2, clusterOntrackStatsEndcapDM2, clusterOfftrackStatsEndcapDM2, rechitStatsEndcapDM2, trackStatsEndcapDM2;
  int digiCounterEndcapDM2, clusterOntrackCounterEndcapDM2, clusterOfftrackCounterEndcapDM2, rechitCounterEndcapDM2, trackCounterEndcapDM2;

  // Digis Cuts:
  MonitorElement * NDigisBarrel;
  MonitorElement * NDigisEndcap;
  MonitorElement * DigiChargeBarrel;
  MonitorElement * DigiChargeEndcap;
  
  // OnTrackCluster Cuts:
  MonitorElement * OnTrackClusterSizeBarrel;
  MonitorElement * OnTrackClusterSizeEndcap;
  MonitorElement * OnTrackClusterChargeBarrel;
  MonitorElement * OnTrackClusterChargeEndcap;
  MonitorElement * OnTrackNClustersBarrel;
  MonitorElement * OnTrackNClustersEndcap;
  
  // OffTrackCluster Cuts:
  MonitorElement * OffTrackClusterSizeBarrel;
  MonitorElement * OffTrackClusterSizeEndcap;
  MonitorElement * OffTrackClusterChargeBarrel;
  MonitorElement * OffTrackClusterChargeEndcap;
  MonitorElement * OffTrackNClustersBarrel;
  MonitorElement * OffTrackNClustersEndcap;
  
  // Residual Cuts:
  MonitorElement * ResidualXMeanBarrel;
  MonitorElement * ResidualXMeanEndcap;
  MonitorElement * ResidualXRMSBarrel;
  MonitorElement * ResidualXRMSEndcap;
  MonitorElement * ResidualYMeanBarrel;
  MonitorElement * ResidualYMeanEndcap;
  MonitorElement * ResidualYRMSBarrel;
  MonitorElement * ResidualYRMSEndcap;
  
  // RechitError Cuts:
  MonitorElement * RecHitErrorXBarrel;
  MonitorElement * RecHitErrorYBarrel;
  MonitorElement * RecHitErrorXEndcap;
  MonitorElement * RecHitErrorYEndcap;
  
};
#endif
