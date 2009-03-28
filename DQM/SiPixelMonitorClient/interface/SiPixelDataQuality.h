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
#include "qstring.h"

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
				 bool                                     Tier0Flag);

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
  
  
  TH2F * allmodsEtaPhi;
  TH2F * errmodsEtaPhi;
  TH2F * goodmodsEtaPhi;
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
  
  // FEDErrors Cuts:
  MonitorElement * NErrorsBarrel;
  MonitorElement * NErrorsEndcap;
  MonitorElement * NErrorsFEDs;
  int n_errors_barrel_, n_errors_endcap_, n_errors_feds_;
  float barrel_error_flag_, endcap_error_flag_, feds_error_flag_;
  
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
  MonitorElement * RecHitErrorsBarrel;
  MonitorElement * RecHitErrorsEndcap;
  
};
#endif
