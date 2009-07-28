/*! \file SiPixelDataQuality.cc
 *  \brief This class represents ...
 *  
 *  (Documentation under development)
 *  
 */
#include "DQM/SiPixelMonitorClient/interface/SiPixelDataQuality.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelEDAClient.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelHistoPlotter.h"
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/TrackerCommon/interface/CgiReader.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "TClass.h"
#include "TText.h"
#include "TROOT.h"
#include "TPad.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TString.h"
#include "TImage.h"
#include "TPaveText.h"
#include "TImageDump.h"
#include "TRandom.h"
#include "TStopwatch.h"
#include "TAxis.h"
#include "TPaveLabel.h"
#include "Rtypes.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"

#include <iostream>
#include <math.h>
#include <map>

#include <fstream>
#include <cstdlib> // for free() - Root can allocate with malloc() - sigh...
 
using namespace std;
using namespace edm;

//------------------------------------------------------------------------------
/*! \brief Constructor of the SiPixelInformationExtractor class.
 *  
 */
SiPixelDataQuality::SiPixelDataQuality(bool offlineXMLfile) : offlineXMLfile_(offlineXMLfile) {
  edm::LogInfo("SiPixelDataQuality") << 
    " Creating SiPixelDataQuality " << "\n" ;
  
  allMods_=0;
  errorMods_=0;
  qflag_=1.;
}

//------------------------------------------------------------------------------
/*! \brief Destructor of the SiPixelDataQuality class.
 *  
 */
SiPixelDataQuality::~SiPixelDataQuality() {
  edm::LogInfo("SiPixelDataQuality") << 
    " Deleting SiPixelDataQuality " << "\n" ;
}


//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *  
 *  Given a pointer to ME returns the associated detId 
 */
int SiPixelDataQuality::getDetId(MonitorElement * mE) 
{
 string mEName = mE->getName() ;

 int detId = 0;
 
 if( mEName.find("_3") != string::npos )
 {
  string detIdString = mEName.substr((mEName.find_last_of("_"))+1,9);
  std::istringstream isst;
  isst.str(detIdString);
  isst>>detId;
// } else {
//  cout << ACYellow << ACBold
//       << "[SiPixelInformationExtractor::getDetId()] "
//       << ACPlain
//       << "Could not extract detId from "
//       << mEName
//       << endl ;
 }
      
  return detId ;
  
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

void SiPixelDataQuality::bookGlobalQualityFlag(DQMStore * bei, bool Tier0Flag) {
//std::cout<<"BOOK GLOBAL QUALITY FLAG MEs!"<<std::endl;
  bei->cd();
  
  bei->setCurrentFolder("Pixel/EventInfo");
  SummaryReport = bei->bookFloat("reportSummary");
  if(!Tier0Flag){
    SummaryReportMap = bei->book2D("reportSummaryMap","Pixel Summary Map",40,0.,40.,36,1.,37.);
    SummaryReportMap->setAxisTitle("Pixel FED #",1);
    SummaryReportMap->setAxisTitle("Pixel FED Channel #",2);
  }else{
    SummaryReportMap = bei->book2D("reportSummaryMap","Pixel Summary Map",7,0.,7.,15,0.,15.);
    SummaryReportMap->setBinLabel(1,"Barrel_Layer_1",1);
    SummaryReportMap->setBinLabel(2,"Barrel_Layer_2",1);
    SummaryReportMap->setBinLabel(3,"Barrel_Layer_3",1);
    SummaryReportMap->setBinLabel(4,"Endcap_Disk_1 -z",1);
    SummaryReportMap->setBinLabel(5,"Endcap_Disk_2 -z",1);
    SummaryReportMap->setBinLabel(6,"Endcap_Disk_1 +z",1);
    SummaryReportMap->setBinLabel(7,"Endcap_Disk_2 +z",1);
    SummaryReportMap->setBinLabel(1,"No errors",2);
    SummaryReportMap->setBinLabel(2,"Pass ndigis cut",2);
    SummaryReportMap->setBinLabel(3,"Pass digi charge cut",2);
    SummaryReportMap->setBinLabel(4,"Pass OnTrack cluster size cut",2);
    SummaryReportMap->setBinLabel(5,"Pass OnTrack nclusters cut",2);
    SummaryReportMap->setBinLabel(6,"Pass OnTrack cluster charge",2);
    SummaryReportMap->setBinLabel(7,"Pass OffTrack cluster size cut",2);
    SummaryReportMap->setBinLabel(8,"Pass OffTrack nclusters cut",2);
    SummaryReportMap->setBinLabel(9,"Pass OffTrack cluster charge",2);
    SummaryReportMap->setBinLabel(10,"Pass residualX_mean cut",2);
    SummaryReportMap->setBinLabel(11,"Pass residualX_RMS cut",2);
    SummaryReportMap->setBinLabel(12,"Pass residualY_mean cut",2);
    SummaryReportMap->setBinLabel(13,"Pass residualY_RMS cut",2);
    SummaryReportMap->setBinLabel(14,"Pass rechit errorX cut",2);
    SummaryReportMap->setBinLabel(15,"Pass rechit errorY cut",2);
  }  
  bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents");
    SummaryPixel = bei->bookFloat("PixelDqmFraction");
    SummaryBarrel = bei->bookFloat("PixelBarrelDqmFraction");
    SummaryEndcap = bei->bookFloat("PixelEndcapDqmFraction");
  // book the data certification cuts:
  bei->setCurrentFolder("Pixel/AdditionalPixelErrors");
    NErrorsFEDs = bei->bookFloat("FEDsNErrorsCut");
  bei->setCurrentFolder("Pixel/Barrel");
    NErrorsBarrel = bei->bookFloat("BarrelNErrorsCut");
    NDigisBarrel = bei->bookInt("BarrelNDigisCut");
    DigiChargeBarrel = bei->bookInt("BarrelDigiChargeCut");
    OnTrackClusterSizeBarrel = bei->bookInt("BarrelOnTrackClusterSizeCut");
    OnTrackNClustersBarrel = bei->bookInt("BarrelOnTrackNClustersCut");
    OnTrackClusterChargeBarrel = bei->bookInt("BarrelOnTrackClusterChargeCut");
    OffTrackClusterSizeBarrel = bei->bookInt("BarrelOffTrackClusterSizeCut");
    OffTrackNClustersBarrel = bei->bookInt("BarrelOffTrackNClustersCut");
    OffTrackClusterChargeBarrel = bei->bookInt("BarrelOffTrackClusterChargeCut");
    ResidualXMeanBarrel = bei->bookInt("BarrelResidualXMeanCut");
    ResidualXRMSBarrel = bei->bookInt("BarrelResidualXRMSCut");
    ResidualYMeanBarrel = bei->bookInt("BarrelResidualYMeanCut");
    ResidualYRMSBarrel = bei->bookInt("BarrelResidualYRMSCut");
    RecHitErrorXBarrel = bei->bookInt("BarrelRecHitErrorXCut");
    RecHitErrorYBarrel = bei->bookInt("BarrelRecHitErrorYCut");
  bei->setCurrentFolder("Pixel/Endcap");
    NErrorsEndcap = bei->bookFloat("EndcapNErrorsCut");
    NDigisEndcap = bei->bookInt("EndcapNDigisCut");
    DigiChargeEndcap = bei->bookInt("EndcapDigiChargeCut");
    OnTrackClusterSizeEndcap = bei->bookInt("EndcapOnTrackClusterSizeCut");
    OnTrackNClustersEndcap = bei->bookInt("EndcapOnTrackNClustersCut");
    OnTrackClusterChargeEndcap = bei->bookInt("EndcapOnTrackClusterChargeCut");
    OffTrackClusterSizeEndcap = bei->bookInt("EndcapOffTrackClusterSizeCut");
    OffTrackNClustersEndcap = bei->bookInt("EndcapOffTrackNClustersCut");
    OffTrackClusterChargeEndcap = bei->bookInt("EndcapOffTrackClusterChargeCut");
    ResidualXMeanEndcap = bei->bookInt("EndcapResidualXMeanCut");
    ResidualXRMSEndcap = bei->bookInt("EndcapResidualXRMSCut");
    ResidualYMeanEndcap = bei->bookInt("EndcapResidualYMeanCut");
    ResidualYRMSEndcap = bei->bookInt("EndcapResidualYRMSCut");
    RecHitErrorXEndcap = bei->bookInt("EndcapRecHitErrorXCut");
    RecHitErrorYEndcap = bei->bookInt("EndcapRecHitErrorYCut");
    
    // Init MonitoringElements:
    SummaryReport = bei->get("Pixel/EventInfo/reportSummary");
    if(SummaryReport) SummaryReport->Fill(1.);
    SummaryPixel = bei->get("Pixel/EventInfo/reportSummaryContents/PixelDqmFraction");
    if(SummaryPixel) SummaryPixel->Fill(1.);
    SummaryBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/PixelBarrelDqmFraction");
    if(SummaryBarrel) SummaryBarrel->Fill(1.);
    SummaryEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/PixelEndcapDqmFraction");
    if(SummaryEndcap)	SummaryEndcap->Fill(1.);
    NErrorsBarrel = bei->get("Pixel/Barrel/BarrelNErrorsCut");
    if(NErrorsBarrel) NErrorsBarrel->Fill(1.);
    NErrorsEndcap = bei->get("Pixel/Endcap/EndcapNErrorsCut");
    if(NErrorsEndcap) NErrorsEndcap->Fill(1.);
    NErrorsFEDs = bei->get("Pixel/AdditionalPixelErrors/FEDsNErrorsCut");
    if(NErrorsFEDs) NErrorsFEDs->Fill(1.);
    NDigisBarrel = bei->get("Pixel/Barrel/BarrelNDigisCut");
    if(NDigisBarrel) NDigisBarrel->Fill(1);
    NDigisEndcap = bei->get("Pixel/Endcap/EndcapNDigisCut");
    if(NDigisEndcap) NDigisEndcap->Fill(1);
    DigiChargeBarrel = bei->get("Pixel/Barrel/BarrelDigiChargeCut");
    if(DigiChargeBarrel) DigiChargeBarrel->Fill(1);
    DigiChargeEndcap = bei->get("Pixel/Endcap/EndcapDigiChargeCut");
    if(DigiChargeEndcap) DigiChargeEndcap->Fill(1);
    OnTrackClusterSizeBarrel = bei->get("Pixel/Barrel/BarrelOnTrackClusterSizeCut");
    if(OnTrackClusterSizeBarrel) OnTrackClusterSizeBarrel->Fill(1);
    OnTrackClusterSizeEndcap = bei->get("Pixel/Endcap/EndcapOnTrackClusterSizeCut");
    if(OnTrackClusterSizeEndcap) OnTrackClusterSizeEndcap->Fill(1);
    OnTrackClusterChargeBarrel = bei->get("Pixel/Barrel/BarrelOnTrackClusterChargeCut");
    if(OnTrackClusterChargeBarrel) OnTrackClusterChargeBarrel->Fill(1);
    OnTrackClusterChargeEndcap = bei->get("Pixel/Endcap/EndcapOnTrackClusterChargeCut");
    if(OnTrackClusterChargeEndcap) OnTrackClusterChargeEndcap->Fill(1);
    OnTrackNClustersBarrel = bei->get("Pixel/Barrel/BarrelOnTrackNClustersCut");
    if(OnTrackNClustersBarrel) OnTrackNClustersBarrel->Fill(1);
    OnTrackNClustersEndcap = bei->get("Pixel/Endcap/EndcapOnTrackNClustersCut");
    if(OnTrackNClustersEndcap) OnTrackNClustersEndcap->Fill(1);
    OffTrackClusterSizeBarrel = bei->get("Pixel/Barrel/BarrelOffTrackClusterSizeCut");
    if(OffTrackClusterSizeBarrel) OffTrackClusterSizeBarrel->Fill(1);
    OffTrackClusterSizeEndcap = bei->get("Pixel/Endcap/EndcapOffTrackClusterSizeCut");
    if(OffTrackClusterSizeEndcap) OffTrackClusterSizeEndcap->Fill(1);
    OffTrackClusterChargeBarrel = bei->get("Pixel/Barrel/BarrelOffTrackClusterChargeCut");
    if(OffTrackClusterChargeBarrel) OffTrackClusterChargeBarrel->Fill(1);
    OffTrackClusterChargeEndcap = bei->get("Pixel/Endcap/EndcapOffTrackClusterChargeCut");
    if(OffTrackClusterChargeEndcap) OffTrackClusterChargeEndcap->Fill(1);
    OffTrackNClustersBarrel = bei->get("Pixel/Barrel/BarrelOffTrackNClustersCut");
    if(OffTrackNClustersBarrel) OffTrackNClustersBarrel->Fill(1);
    OffTrackNClustersEndcap = bei->get("Pixel/Endcap/EndcapOffTrackNClustersCut");
    if(OffTrackNClustersEndcap) OffTrackNClustersEndcap->Fill(1);
    ResidualXMeanBarrel = bei->get("Pixel/Barrel/BarrelResidualXMeanCut");
    if(ResidualXMeanBarrel) ResidualXMeanBarrel->Fill(1);
    ResidualXMeanEndcap = bei->get("Pixel/Endcap/EndcapResidualXMeanCut");
    if(ResidualXMeanEndcap) ResidualXMeanEndcap->Fill(1);
    ResidualXRMSBarrel = bei->get("Pixel/Barrel/BarrelResidualXRMSCut");
    if(ResidualXRMSBarrel) ResidualXRMSBarrel->Fill(1);
    ResidualXRMSEndcap = bei->get("Pixel/Endcap/EndcapResidualXRMSCut");
    if(ResidualXRMSEndcap) ResidualXRMSEndcap->Fill(1);
    ResidualYMeanBarrel = bei->get("Pixel/Barrel/BarrelResidualYMeanCut");
    if(ResidualYMeanBarrel) ResidualYMeanBarrel->Fill(1);
    ResidualYMeanEndcap = bei->get("Pixel/Endcap/EndcapResidualYMeanCut");
    if(ResidualYMeanEndcap) ResidualYMeanEndcap->Fill(1);
    ResidualYRMSBarrel = bei->get("Pixel/Barrel/BarrelResidualYRMSCut");
    if(ResidualYRMSBarrel) ResidualYRMSBarrel->Fill(1);
    ResidualYRMSEndcap = bei->get("Pixel/Endcap/EndcapResidualYRMSCut");
    if(ResidualYRMSEndcap) ResidualYRMSEndcap->Fill(1);
    RecHitErrorXBarrel = bei->get("Pixel/Barrel/BarrelRecHitErrorXCut");
    if(RecHitErrorXBarrel) RecHitErrorXBarrel->Fill(1);
    RecHitErrorYBarrel = bei->get("Pixel/Barrel/BarrelRecHitErrorYCut");
    if(RecHitErrorYBarrel) RecHitErrorYBarrel->Fill(1);
    RecHitErrorXEndcap = bei->get("Pixel/Endcap/EndcapRecHitErrorXCut");
    if(RecHitErrorXEndcap) RecHitErrorXEndcap->Fill(1);
    RecHitErrorYEndcap = bei->get("Pixel/Endcap/EndcapRecHitErrorYCut");
    if(RecHitErrorYEndcap) RecHitErrorYEndcap->Fill(1);
  bei->cd();  
}

//**********************************************************************************************

void SiPixelDataQuality::computeGlobalQualityFlag(DQMStore * bei, 
                                                           bool init,
							   int nFEDs,
							   bool Tier0Flag)
{
//cout<<"entering SiPixelDataQuality::ComputeGlobalQualityFlag"<<endl;
//   cout << ACRed << ACBold
//        << "[SiPixelDataQuality::ComputeGlobalQualityFlag]"
//        << ACPlain
//        << " Enter" 
//        << endl ;
  if(init){
    allMods_=0; errorMods_=0; qflag_=0.; 
    barrelMods_=0; endcapMods_=0;
    
    objectCount_=0;
    DONE_ = false;
    
    //Error counters and flags:
    n_errors_barrel_=0; barrel_error_flag_=0.;
    n_errors_endcap_=0; endcap_error_flag_=0.;
    n_errors_feds_=0; feds_error_flag_=0.;
    n_errors_barrelL1_=0; n_errors_barrelL2_=0; n_errors_barrelL3_=0; 
    n_errors_endcapDP1_=0; n_errors_endcapDP2_=0; n_errors_endcapDM1_=0; n_errors_endcapDM2_=0;
    BarrelL1_error_flag_=-1.; BarrelL2_error_flag_=-1.; BarrelL3_error_flag_=-1.;
    EndcapDP1_error_flag_=-1.; EndcapDP2_error_flag_=-1.; EndcapDM1_error_flag_=-1.; EndcapDM2_error_flag_=-1.; 
    for(int i=0; i!=14; i++){
      BarrelL1_cuts_flag_[i]=-1.; BarrelL2_cuts_flag_[i]=-1.; BarrelL3_cuts_flag_[i]=-1.;
      EndcapDP1_cuts_flag_[i]=-1.; EndcapDP2_cuts_flag_[i]=-1.; EndcapDM1_cuts_flag_[i]=-1.;
      EndcapDM2_cuts_flag_[i]=-1.; 
    }
    init=false;
  }
  if(nFEDs==0) return;  
  
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);

  bool digiStatsBarrel = false, clusterOntrackStatsBarrel = false, clusterOfftrackStatsBarrel = false, rechitStatsBarrel = false, trackStatsBarrel = false;
  int digiCounterBarrel = 0, clusterOntrackCounterBarrel = 0, clusterOfftrackCounterBarrel = 0, rechitCounterBarrel = 0, trackCounterBarrel = 0;
  bool digiStatsEndcap = false, clusterOntrackStatsEndcap = false, clusterOfftrackStatsEndcap = false, rechitStatsEndcap = false, trackStatsEndcap = false;
  int digiCounterEndcap = 0, clusterOntrackCounterEndcap = 0, clusterOfftrackCounterEndcap = 0, rechitCounterEndcap = 0, trackCounterEndcap = 0;
  bool digiStatsBarrelL1 = false, clusterOntrackStatsBarrelL1 = false, clusterOfftrackStatsBarrelL1 = false, rechitStatsBarrelL1 = false, trackStatsBarrelL1 = false;
  int digiCounterBarrelL1 = 0, clusterOntrackCounterBarrelL1 = 0, clusterOfftrackCounterBarrelL1 = 0, rechitCounterBarrelL1 = 0, trackCounterBarrelL1 = 0;
  bool digiStatsBarrelL2 = false, clusterOntrackStatsBarrelL2 = false, clusterOfftrackStatsBarrelL2 = false, rechitStatsBarrelL2 = false, trackStatsBarrelL2 = false;
  int digiCounterBarrelL2 = 0, clusterOntrackCounterBarrelL2 = 0, clusterOfftrackCounterBarrelL2 = 0, rechitCounterBarrelL2 = 0, trackCounterBarrelL2 = 0;
  bool digiStatsBarrelL3 = false, clusterOntrackStatsBarrelL3 = false, clusterOfftrackStatsBarrelL3 = false, rechitStatsBarrelL3 = false, trackStatsBarrelL3 = false;
  int digiCounterBarrelL3 = 0, clusterOntrackCounterBarrelL3 = 0, clusterOfftrackCounterBarrelL3 = 0, rechitCounterBarrelL3 = 0, trackCounterBarrelL3 = 0;
  bool digiStatsEndcapDP1 = false, clusterOntrackStatsEndcapDP1 = false, clusterOfftrackStatsEndcapDP1 = false, rechitStatsEndcapDP1 = false, trackStatsEndcapDP1 = false;
  int digiCounterEndcapDP1 = 0, clusterOntrackCounterEndcapDP1 = 0, clusterOfftrackCounterEndcapDP1 = 0, rechitCounterEndcapDP1 = 0, trackCounterEndcapDP1 = 0;
  bool digiStatsEndcapDP2 = false, clusterOntrackStatsEndcapDP2 = false, clusterOfftrackStatsEndcapDP2 = false, rechitStatsEndcapDP2 = false, trackStatsEndcapDP2 = false;
  int digiCounterEndcapDP2 = 0, clusterOntrackCounterEndcapDP2 = 0, clusterOfftrackCounterEndcapDP2 = 0, rechitCounterEndcapDP2 = 0, trackCounterEndcapDP2 = 0;
  bool digiStatsEndcapDM1 = false, clusterOntrackStatsEndcapDM1 = false, clusterOfftrackStatsEndcapDM1 = false, rechitStatsEndcapDM1 = false, trackStatsEndcapDM1 = false;
  int digiCounterEndcapDM1 = 0, clusterOntrackCounterEndcapDM1 = 0, clusterOfftrackCounterEndcapDM1 = 0, rechitCounterEndcapDM1 = 0, trackCounterEndcapDM1 = 0;
  bool digiStatsEndcapDM2 = false, clusterOntrackStatsEndcapDM2 = false, clusterOfftrackStatsEndcapDM2 = false, rechitStatsEndcapDM2 = false, trackStatsEndcapDM2 = false;
  int digiCounterEndcapDM2 = 0, clusterOntrackCounterEndcapDM2 = 0, clusterOfftrackCounterEndcapDM2 = 0, rechitCounterEndcapDM2 = 0, trackCounterEndcapDM2 = 0;
  
  if((!Tier0Flag && dname.find("Module_")!=string::npos) || 
     (Tier0Flag && (dname.find("Ladder_")!=string::npos || dname.find("Blade_")!=string::npos))){

    objectCount_++;

    if(currDir.find("Pixel")!=string::npos) allMods_++;
    if(currDir.find("Barrel")!=string::npos) barrelMods_++;
    if(currDir.find("Endcap")!=string::npos) endcapMods_++;
    if(currDir.find("Layer_1")!=string::npos) barrelModsL1_++;
    if(currDir.find("Layer_2")!=string::npos) barrelModsL2_++;
    if(currDir.find("Layer_3")!=string::npos) barrelModsL3_++;
    if(currDir.find("HalfCylinder_pI/Disk_1")!=string::npos || currDir.find("HalfCylinder_pO/Disk_1")!=string::npos) endcapModsDP1_++;
    if(currDir.find("HalfCylinder_pI/Disk_2")!=string::npos || currDir.find("HalfCylinder_pO/Disk_2")!=string::npos) endcapModsDP2_++;
    if(currDir.find("HalfCylinder_mI/Disk_1")!=string::npos || currDir.find("HalfCylinder_mO/Disk_1")!=string::npos) endcapModsDM1_++;
    if(currDir.find("HalfCylinder_mI/Disk_2")!=string::npos || currDir.find("HalfCylinder_mO/Disk_2")!=string::npos) endcapModsDM2_++;
      
    vector<string> meVec = bei->getMEs();
    for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
      string full_path = currDir + "/" + (*it);
      if(full_path.find("_NErrors_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>0){
	  full_path = full_path.replace(full_path.find("NErrors"),9,"errorType");
	  me = bei->get(full_path);
	  if(!me) continue;
	  bool type30=false; bool othererror=false; bool reset=false;
	  for(int jj=1; jj<16; jj++){
	    if(me->getBinContent(jj)>0.){
	      if(jj!=6) othererror=true;
              else type30=true;
	    }
	  }
	  if(type30){
	    full_path = full_path.replace(full_path.find("errorType"),10,"TBMMessage");
	    me = bei->get(full_path);
	    if(!me) continue;
	    for(int kk=1; kk<9; kk++){
              if(me->getBinContent(kk)>0.){
		if(kk!=6 && kk!=7) othererror=true;
		else reset=true;
	      }
	    }
	  }
	  if(othererror || (type30 && !reset)){
            if(currDir.find("Pixel")!=string::npos) errorMods_++;
            if(currDir.find("Barrel")!=string::npos) n_errors_barrel_++;
            if(currDir.find("Endcap")!=string::npos) n_errors_endcap_++;
            if(currDir.find("Layer_1")!=string::npos) n_errors_barrelL1_++;
            if(currDir.find("Layer_2")!=string::npos) n_errors_barrelL2_++;
            if(currDir.find("Layer_3")!=string::npos) n_errors_barrelL3_++;
            if(currDir.find("HalfCylinder_pI/Disk_1")!=string::npos || currDir.find("HalfCylinder_pO/Disk_1")!=string::npos) n_errors_endcapDP1_++;
            if(currDir.find("HalfCylinder_pI/Disk_2")!=string::npos || currDir.find("HalfCylinder_pO/Disk_2")!=string::npos) n_errors_endcapDP2_++;
            if(currDir.find("HalfCylinder_mI/Disk_1")!=string::npos || currDir.find("HalfCylinder_mO/Disk_1")!=string::npos) n_errors_endcapDM1_++;
            if(currDir.find("HalfCylinder_mI/Disk_2")!=string::npos || currDir.find("HalfCylinder_mO/Disk_2")!=string::npos) n_errors_endcapDM2_++;
	  }
        }	
      }else if(full_path.find("_ndigis_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>50){
	  if(full_path.find("Barrel")!=string::npos) digiCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) digiCounterEndcap++;
          if(full_path.find("Layer_1")!=string::npos) digiCounterBarrelL1++;
          if(full_path.find("Layer_2")!=string::npos) digiCounterBarrelL2++;
          if(full_path.find("Layer_3")!=string::npos) digiCounterBarrelL3++;
          if(full_path.find("HalfCylinder_pI/Disk_1")!=string::npos || currDir.find("HalfCylinder_pO/Disk_1")!=string::npos) digiCounterEndcapDP1++;
          if(full_path.find("HalfCylinder_pI/Disk_2")!=string::npos || currDir.find("HalfCylinder_pO/Disk_2")!=string::npos) digiCounterEndcapDP2++;
          if(full_path.find("HalfCylinder_mI/Disk_1")!=string::npos || currDir.find("HalfCylinder_mO/Disk_1")!=string::npos) digiCounterEndcapDM1++;
          if(full_path.find("HalfCylinder_mI/Disk_2")!=string::npos || currDir.find("HalfCylinder_mO/Disk_2")!=string::npos) digiCounterEndcapDM2++;
        }
      }else if(full_path.find("_nclusters_OnTrack_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>50){
	  if(full_path.find("Barrel")!=string::npos) clusterOntrackCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) clusterOntrackCounterEndcap++;
          if(full_path.find("Layer_1")!=string::npos) clusterOntrackCounterBarrelL1++;
          if(full_path.find("Layer_2")!=string::npos) clusterOntrackCounterBarrelL2++;
          if(full_path.find("Layer_3")!=string::npos) clusterOntrackCounterBarrelL3++;
          if(full_path.find("HalfCylinder_pI/Disk_1")!=string::npos || currDir.find("HalfCylinder_pO/Disk_1")!=string::npos) clusterOntrackCounterEndcapDP1++;
          if(full_path.find("HalfCylinder_pI/Disk_2")!=string::npos || currDir.find("HalfCylinder_pO/Disk_2")!=string::npos) clusterOntrackCounterEndcapDP2++;
          if(full_path.find("HalfCylinder_mI/Disk_1")!=string::npos || currDir.find("HalfCylinder_mO/Disk_1")!=string::npos) clusterOntrackCounterEndcapDM1++;
          if(full_path.find("HalfCylinder_mI/Disk_2")!=string::npos || currDir.find("HalfCylinder_mO/Disk_2")!=string::npos) clusterOntrackCounterEndcapDM2++;
        }
      }else if(full_path.find("_nclusters_OffTrack_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>50){
	  if(full_path.find("Barrel")!=string::npos) clusterOfftrackCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) clusterOfftrackCounterEndcap++;
          if(full_path.find("Layer_1")!=string::npos) clusterOfftrackCounterBarrelL1++;
          if(full_path.find("Layer_2")!=string::npos) clusterOfftrackCounterBarrelL2++;
          if(full_path.find("Layer_3")!=string::npos) clusterOfftrackCounterBarrelL3++;
          if(full_path.find("HalfCylinder_pI/Disk_1")!=string::npos || currDir.find("HalfCylinder_pO/Disk_1")!=string::npos) clusterOfftrackCounterEndcapDP1++;
          if(full_path.find("HalfCylinder_pI/Disk_2")!=string::npos || currDir.find("HalfCylinder_pO/Disk_2")!=string::npos) clusterOfftrackCounterEndcapDP2++;
          if(full_path.find("HalfCylinder_mI/Disk_1")!=string::npos || currDir.find("HalfCylinder_mO/Disk_1")!=string::npos) clusterOfftrackCounterEndcapDM1++;
          if(full_path.find("HalfCylinder_mI/Disk_2")!=string::npos || currDir.find("HalfCylinder_mO/Disk_2")!=string::npos) clusterOfftrackCounterEndcapDM2++;
        }
      }else if(full_path.find("_nRecHits_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>50){
	  if(full_path.find("Barrel")!=string::npos) rechitCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) rechitCounterEndcap++;
          if(full_path.find("Layer_1")!=string::npos) rechitCounterBarrelL1++;
          if(full_path.find("Layer_2")!=string::npos) rechitCounterBarrelL2++;
          if(full_path.find("Layer_3")!=string::npos) rechitCounterBarrelL3++;
          if(full_path.find("HalfCylinder_pI/Disk_1")!=string::npos || currDir.find("HalfCylinder_pO/Disk_1")!=string::npos) rechitCounterEndcapDP1++;
          if(full_path.find("HalfCylinder_pI/Disk_2")!=string::npos || currDir.find("HalfCylinder_pO/Disk_2")!=string::npos) rechitCounterEndcapDP2++;
          if(full_path.find("HalfCylinder_mI/Disk_1")!=string::npos || currDir.find("HalfCylinder_mO/Disk_1")!=string::npos) rechitCounterEndcapDM1++;
          if(full_path.find("HalfCylinder_mI/Disk_2")!=string::npos || currDir.find("HalfCylinder_mO/Disk_2")!=string::npos) rechitCounterEndcapDM2++;
        }
      }else if(full_path.find("_residualX_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>50){
	  if(full_path.find("Barrel")!=string::npos) trackCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) trackCounterEndcap++;
          if(full_path.find("Layer_1")!=string::npos) trackCounterBarrelL1++;
          if(full_path.find("Layer_2")!=string::npos) trackCounterBarrelL2++;
          if(full_path.find("Layer_3")!=string::npos) trackCounterBarrelL3++;
          if(full_path.find("HalfCylinder_pI/Disk_1")!=string::npos || currDir.find("HalfCylinder_pO/Disk_1")!=string::npos) trackCounterEndcapDP1++;
          if(full_path.find("HalfCylinder_pI/Disk_2")!=string::npos || currDir.find("HalfCylinder_pO/Disk_2")!=string::npos) trackCounterEndcapDP2++;
          if(full_path.find("HalfCylinder_mI/Disk_1")!=string::npos || currDir.find("HalfCylinder_mO/Disk_1")!=string::npos) trackCounterEndcapDM1++;
          if(full_path.find("HalfCylinder_mI/Disk_2")!=string::npos || currDir.find("HalfCylinder_mO/Disk_2")!=string::npos) trackCounterEndcapDM2++;
        }
      }
    }
  }
  vector<string> subDirVec = bei->getSubdirs();  
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    bei->cd(*ic);
    init=false;
    computeGlobalQualityFlag(bei,init,nFEDs,Tier0Flag);
    bei->goUp();
  }
  
  // Make sure I have finished looping over all Modules/Ladders/Blades:
  if(!Tier0Flag){ // online case
    if(objectCount_ == 1440) DONE_ = true;
  }else{ // offline case
    if(objectCount_ == 288) DONE_ = true;
  } 
  
  if(DONE_ && currDir=="Pixel/EventInfo/reportSummaryContents"){ 
  // Fill the FED Error flags:
  if(barrelMods_>0) barrel_error_flag_ = (float(barrelMods_)-float(n_errors_barrel_))/float(barrelMods_);
  if(endcapMods_>0) endcap_error_flag_ = (float(endcapMods_)-float(n_errors_endcap_))/float(endcapMods_);
  if(barrelModsL1_>0) BarrelL1_error_flag_ = (float(barrelModsL1_)-float(n_errors_barrelL1_))/float(barrelModsL1_);
  if(barrelModsL2_>0) BarrelL2_error_flag_ = (float(barrelModsL2_)-float(n_errors_barrelL2_))/float(barrelModsL2_);
  if(barrelModsL3_>0) BarrelL3_error_flag_ = (float(barrelModsL3_)-float(n_errors_barrelL3_))/float(barrelModsL3_);
  if(endcapModsDP1_>0) EndcapDP1_error_flag_ = (float(endcapModsDP1_)-float(n_errors_endcapDP1_))/float(endcapModsDP1_);
  if(endcapModsDP2_>0) EndcapDP2_error_flag_ = (float(endcapModsDP2_)-float(n_errors_endcapDP2_))/float(endcapModsDP2_);
  if(endcapModsDM1_>0) EndcapDM1_error_flag_ = (float(endcapModsDM1_)-float(n_errors_endcapDM1_))/float(endcapModsDM1_);
  if(endcapModsDM2_>0) EndcapDM2_error_flag_ = (float(endcapModsDM2_)-float(n_errors_endcapDM2_))/float(endcapModsDM2_);
  NErrorsBarrel = bei->get("Pixel/Barrel/BarrelNErrorsCut");
  if(NErrorsBarrel) NErrorsBarrel->Fill(barrel_error_flag_);
  NErrorsEndcap = bei->get("Pixel/Endcap/EndcapNErrorsCut");
  if(NErrorsEndcap)   NErrorsEndcap->Fill(endcap_error_flag_);
  NErrorsFEDs = bei->get("Pixel/AdditionalPixelErrors/FEDsNErrorsCut");
  if(NErrorsFEDs) NErrorsFEDs->Fill(1.); // hardwired for the moment, need to fix!

  
  string meName0, meName1, meName2, meName3, meName4, meName5, meName6, meName7, meName8; 
  string meName9, meName10, meName11, meName12, meName13, meName14, meName15, meName16;
  string meName17, meName18, meName19, meName20;
  MonitorElement * me, * me1, * me2, * me3, * me4;
  
  // Fill the Digi flags:
  if(!Tier0Flag){
    meName0 = "Pixel/Barrel/SUMDIG_ndigis_Barrel";
    if(digiCounterBarrel/768 > 0.9) digiStatsBarrel = true;
    if(digiCounterEndcap/672 > 0.9) digiStatsEndcap = true;
    //cout<<"digiStatsBarrel="<<digiStatsBarrel<<" , digiStatsEndcap="<<digiStatsEndcap<<endl;
  }else{
    meName0 = "Pixel/Barrel/SUMOFF_ndigis_Barrel"; 
    if(digiCounterBarrel/192 > 0.9) digiStatsBarrel = true;
    if(digiCounterEndcap/96 > 0.9) digiStatsEndcap = true;
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_ndigis_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_ndigis_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_ndigis_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_ndigis_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_ndigis_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_ndigis_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_ndigis_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_ndigis_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_ndigis_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_ndigis_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_ndigis_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_ndigis_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_ndigis_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_ndigis_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_ndigis_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_ndigis_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_ndigis_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_ndigis_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_ndigis_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_ndigis_Disk_2"; 
    if(digiCounterBarrelL1/40 > 0.9) digiStatsBarrelL1 = true;
    if(digiCounterBarrelL2/64 > 0.9) digiStatsBarrelL2 = true;
    if(digiCounterBarrelL3/88 > 0.9) digiStatsBarrelL3 = true;
    if(digiCounterEndcapDP1/24 > 0.9) digiStatsEndcapDP1 = true;
    if(digiCounterEndcapDP2/24 > 0.9) digiStatsEndcapDP2 = true;
    if(digiCounterEndcapDM1/24 > 0.9) digiStatsEndcapDM1 = true;
    if(digiCounterEndcapDM2/24 > 0.9) digiStatsEndcapDM2 = true;
  }
  me = bei->get(meName0);
  if(me){
    NDigisBarrel = bei->get("Pixel/Barrel/BarrelNDigisCut");
   // cout<<"NDigis: "<<NDigisBarrel<<" , "<<digiStatsBarrel<<" , "<<me->hasError()<<endl;
    if(NDigisBarrel && digiStatsBarrel){
      if(me->hasError()) NDigisBarrel->Fill(0);
      else NDigisBarrel->Fill(1); 
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMDIG_ndigis_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_ndigis_Endcap"; 
  me = bei->get(meName0);
  if(me){
    NDigisEndcap = bei->get("Pixel/Endcap/EndcapNDigisCut");
    if(NDigisEndcap && digiStatsEndcap){
      if(me->hasError()) NDigisEndcap->Fill(0);
      else NDigisEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(digiStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[0]=0.;
    else if(digiStatsBarrelL1) BarrelL1_cuts_flag_[0]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(digiStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[0]=0.;
    else if(digiStatsBarrelL2) BarrelL2_cuts_flag_[0]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(digiStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[0]=0.;
    else if(digiStatsBarrelL3) BarrelL3_cuts_flag_[0]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(digiStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[0]=0.;
    else if(digiStatsEndcapDM1) EndcapDM1_cuts_flag_[0]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(digiStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[0]=0.;
    else if(digiStatsEndcapDP1) EndcapDP1_cuts_flag_[0]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(digiStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[0]=0.;
    else if(digiStatsEndcapDM2) EndcapDM2_cuts_flag_[0]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(digiStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[0]=0.;
    else if(digiStatsEndcapDP2) EndcapDP2_cuts_flag_[0]=1.;
  }
  if(!Tier0Flag) meName0 = "Pixel/Barrel/SUMDIG_adc_Barrel";
  else meName0 = "Pixel/Barrel/SUMOFF_adc_Barrel"; 
  me = bei->get(meName0);
  if(me){
    DigiChargeBarrel = bei->get("Pixel/Barrel/BarrelDigiChargeCut");
    if(DigiChargeBarrel && digiStatsBarrel){
      if(me->hasError()) DigiChargeBarrel->Fill(0);
      else DigiChargeBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMDIG_adc_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_adc_Endcap"; 
  me = bei->get(meName0);
  if(me){
    DigiChargeEndcap = bei->get("Pixel/Endcap/EndcapDigiChargeCut");
    if(DigiChargeEndcap && digiStatsEndcap){
      if(me->hasError()) DigiChargeEndcap->Fill(0);
      else DigiChargeEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_adc_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_adc_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_adc_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_adc_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_adc_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_adc_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_adc_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_adc_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_adc_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_adc_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_adc_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_adc_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_adc_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_adc_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_adc_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_adc_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_adc_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_adc_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_adc_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_adc_Disk_2"; 
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(digiStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[1]=0.;
    else if(digiStatsBarrelL1) BarrelL1_cuts_flag_[1]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(digiStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[1]=0.;
    else if(digiStatsBarrelL2) BarrelL2_cuts_flag_[1]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(digiStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[1]=0.;
    else if(digiStatsBarrelL3) BarrelL3_cuts_flag_[1]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(digiStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[1]=0.;
    else if(digiStatsEndcapDM1) EndcapDM1_cuts_flag_[1]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(digiStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[1]=0.;
    else if(digiStatsEndcapDP1) EndcapDP1_cuts_flag_[1]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(digiStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[1]=0.;
    else if(digiStatsEndcapDM2) EndcapDM2_cuts_flag_[1]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(digiStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[1]=0.;
    else if(digiStatsEndcapDP2) EndcapDP2_cuts_flag_[1]=1.;
  }
     
     
    // Fill the OnTrack Cluster flags:
  if(!Tier0Flag){
    meName0 = "Pixel/Barrel/SUMTRK_size_OnTrack_Barrel";
    if(clusterOntrackCounterBarrel/768 > 0.9) clusterOntrackStatsBarrel = true;
    if(clusterOntrackCounterEndcap/672 > 0.9) clusterOntrackStatsEndcap = true;
  }else{  
    meName0 = "Pixel/Barrel/SUMOFF_size_OnTrack_Barrel"; 
    if(clusterOntrackCounterBarrel/192 > 0.9) clusterOntrackStatsBarrel = true;
    if(clusterOntrackCounterEndcap/96 > 0.9) clusterOntrackStatsEndcap = true;
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_size_OnTrack_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_size_OnTrack_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_size_OnTrack_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_size_OnTrack_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_size_OnTrack_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_size_OnTrack_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_size_OnTrack_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_size_OnTrack_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_size_OnTrack_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_size_OnTrack_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_size_OnTrack_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_size_OnTrack_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_size_OnTrack_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_size_OnTrack_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_size_OnTrack_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_size_OnTrack_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_size_OnTrack_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_size_OnTrack_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_size_OnTrack_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_size_OnTrack_Disk_2"; 
    if(clusterOntrackCounterBarrelL1/40 > 0.9)  clusterOntrackStatsBarrelL1 = true;
    if(clusterOntrackCounterBarrelL2/64 > 0.9)  clusterOntrackStatsBarrelL2 = true;
    if(clusterOntrackCounterBarrelL3/88 > 0.9)  clusterOntrackStatsBarrelL3 = true;
    if(clusterOntrackCounterEndcapDP1/24 > 0.9) clusterOntrackStatsEndcapDP1 = true;
    if(clusterOntrackCounterEndcapDP2/24 > 0.9) clusterOntrackStatsEndcapDP2 = true;
    if(clusterOntrackCounterEndcapDM1/24 > 0.9) clusterOntrackStatsEndcapDM1 = true;
    if(clusterOntrackCounterEndcapDM2/24 > 0.9) clusterOntrackStatsEndcapDM2 = true;
  }
  me = bei->get(meName0);
  if(me){
    OnTrackClusterSizeBarrel = bei->get("Pixel/Barrel/BarrelOnTrackClusterSizeCut");
    if(OnTrackClusterSizeBarrel && clusterOntrackStatsBarrel){
      if(me->hasError()) OnTrackClusterSizeBarrel->Fill(0);
      else OnTrackClusterSizeBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMTRK_size_OnTrack_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_size_OnTrack_Endcap"; 
  me = bei->get(meName0);
  if(me){
    OnTrackClusterSizeEndcap = bei->get("Pixel/Endcap/EndcapOnTrackClusterSizeCut");
    if(OnTrackClusterSizeEndcap && clusterOntrackStatsEndcap){
      if(me->hasError()) OnTrackClusterSizeEndcap->Fill(0);
      else OnTrackClusterSizeEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(clusterOntrackStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[2]=0.;
    else if(clusterOntrackStatsBarrelL1) BarrelL1_cuts_flag_[2]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(clusterOntrackStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[2]=0.;
    else if(clusterOntrackStatsBarrelL2) BarrelL2_cuts_flag_[2]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(clusterOntrackStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[2]=0.;
    else if(clusterOntrackStatsBarrelL3) BarrelL3_cuts_flag_[2]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(clusterOntrackStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[2]=0.;
    else if(clusterOntrackStatsEndcapDM1) EndcapDM1_cuts_flag_[2]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(clusterOntrackStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[2]=0.;
    else if(clusterOntrackStatsEndcapDP1) EndcapDP1_cuts_flag_[2]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(clusterOntrackStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[2]=0.;
    else if(clusterOntrackStatsEndcapDM2) EndcapDM2_cuts_flag_[2]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(clusterOntrackStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[2]=0.;
    else if(clusterOntrackStatsEndcapDP2) EndcapDP2_cuts_flag_[2]=1.;
  }
  if(!Tier0Flag) meName0 = "Pixel/Barrel/SUMTRK_charge_OnTrack_Barrel";
  else meName0 = "Pixel/Barrel/SUMOFF_charge_OnTrack_Barrel"; 
  me = bei->get(meName0);
  if(me){
    OnTrackClusterChargeBarrel = bei->get("Pixel/Barrel/BarrelOnTrackClusterChargeCut");
    if(OnTrackClusterChargeBarrel && clusterOntrackStatsBarrel){
      if(me->hasError()) OnTrackClusterChargeBarrel->Fill(0);
      else OnTrackClusterChargeBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMTRK_charge_OnTrack_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_charge_OnTrack_Endcap"; 
  me = bei->get(meName0);
  if(me){
    OnTrackClusterChargeEndcap = bei->get("Pixel/Endcap/EndcapOnTrackClusterChargeCut");
    if(OnTrackClusterChargeEndcap && clusterOntrackStatsEndcap){
      if(me->hasError()) OnTrackClusterChargeEndcap->Fill(0);
      else OnTrackClusterChargeEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_charge_OnTrack_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_charge_OnTrack_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_charge_OnTrack_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_charge_OnTrack_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_charge_OnTrack_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_charge_OnTrack_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_charge_OnTrack_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_charge_OnTrack_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_charge_OnTrack_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_charge_OnTrack_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_charge_OnTrack_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_charge_OnTrack_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_charge_OnTrack_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_charge_OnTrack_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_charge_OnTrack_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_charge_OnTrack_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_charge_OnTrack_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_charge_OnTrack_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_charge_OnTrack_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_charge_OnTrack_Disk_2"; 
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(clusterOntrackStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[3]=0.;
    else if(clusterOntrackStatsBarrelL1) BarrelL1_cuts_flag_[3]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(clusterOntrackStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[3]=0.;
    else if(clusterOntrackStatsBarrelL2) BarrelL2_cuts_flag_[3]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(clusterOntrackStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[3]=0.;
    else if(clusterOntrackStatsBarrelL3) BarrelL3_cuts_flag_[3]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(clusterOntrackStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[3]=0.;
    else if(clusterOntrackStatsEndcapDM1) EndcapDM1_cuts_flag_[3]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(clusterOntrackStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[3]=0.;
    else if(clusterOntrackStatsEndcapDP1) EndcapDP1_cuts_flag_[3]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(clusterOntrackStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[3]=0.;
    else if(clusterOntrackStatsEndcapDM2) EndcapDM2_cuts_flag_[3]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(clusterOntrackStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[3]=0.;
    else if(clusterOntrackStatsEndcapDP2) EndcapDP2_cuts_flag_[3]=1.;
  }
  if(!Tier0Flag) meName0 = "Pixel/Barrel/SUMTRK_nclusters_OnTrack_Barrel";
  else meName0 = "Pixel/Barrel/SUMOFF_nclusters_OnTrack_Barrel"; 
  me = bei->get(meName0);
  if(me){
    OnTrackNClustersBarrel = bei->get("Pixel/Barrel/BarrelOnTrackNClustersCut");
    if(OnTrackNClustersBarrel && clusterOntrackStatsBarrel){
      if(me->hasError()) OnTrackNClustersBarrel->Fill(0);
      else OnTrackNClustersBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMTRK_nclusters_OnTrack_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_nclusters_OnTrack_Endcap"; 
  me = bei->get(meName0);
  if(me){
    OnTrackNClustersEndcap = bei->get("Pixel/Endcap/EndcapOnTrackNClustersCut");
    if(OnTrackNClustersEndcap && clusterOntrackStatsEndcap){
      if(me->hasError()) OnTrackNClustersEndcap->Fill(0);
      else OnTrackNClustersEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_nclusters_OnTrack_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_nclusters_OnTrack_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_nclusters_OnTrack_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_nclusters_OnTrack_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_nclusters_OnTrack_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_nclusters_OnTrack_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_nclusters_OnTrack_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_nclusters_OnTrack_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_nclusters_OnTrack_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_nclusters_OnTrack_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_nclusters_OnTrack_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_nclusters_OnTrack_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_nclusters_OnTrack_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_nclusters_OnTrack_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_nclusters_OnTrack_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_nclusters_OnTrack_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_nclusters_OnTrack_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_nclusters_OnTrack_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_nclusters_OnTrack_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_nclusters_OnTrack_Disk_2"; 
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(clusterOntrackStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[4]=0.;
    else if(clusterOntrackStatsBarrelL1) BarrelL1_cuts_flag_[4]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(clusterOntrackStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[4]=0.;
    else if(clusterOntrackStatsBarrelL2) BarrelL2_cuts_flag_[4]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(clusterOntrackStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[4]=0.;
    else if(clusterOntrackStatsBarrelL3) BarrelL3_cuts_flag_[4]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(clusterOntrackStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[4]=0.;
    else if(clusterOntrackStatsEndcapDM1) EndcapDM1_cuts_flag_[4]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(clusterOntrackStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[4]=0.;
    else if(clusterOntrackStatsEndcapDP1) EndcapDP1_cuts_flag_[4]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(clusterOntrackStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[4]=0.;
    else if(clusterOntrackStatsEndcapDM2) EndcapDM2_cuts_flag_[4]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(clusterOntrackStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[4]=0.;
    else if(clusterOntrackStatsEndcapDP2) EndcapDP2_cuts_flag_[4]=1.;
  }

     
  // Fill the OffTrack Cluster flags:
  if(!Tier0Flag){
    meName0 = "Pixel/Barrel/SUMTRK_size_OffTrack_Barrel";
    if(clusterOfftrackCounterBarrel/768 > 0.9) clusterOfftrackStatsBarrel = true;
    if(clusterOfftrackCounterEndcap/672 > 0.9) clusterOfftrackStatsEndcap = true;
  }else{
    meName0 = "Pixel/Barrel/SUMOFF_size_OffTrack_Barrel"; 
    if(clusterOfftrackCounterBarrel/192 > 0.9) clusterOfftrackStatsBarrel = true;
    if(clusterOfftrackCounterEndcap/96 > 0.9) clusterOfftrackStatsEndcap = true;
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_size_OffTrack_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_size_OffTrack_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_size_OffTrack_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_size_OffTrack_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_size_OffTrack_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_size_OffTrack_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_size_OffTrack_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_size_OffTrack_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_size_OffTrack_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_size_OffTrack_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_size_OffTrack_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_size_OffTrack_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_size_OffTrack_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_size_OffTrack_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_size_OffTrack_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_size_OffTrack_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_size_OffTrack_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_size_OffTrack_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_size_OffTrack_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_size_OffTrack_Disk_2"; 
    if(clusterOfftrackCounterBarrelL1/40 > 0.9)  clusterOfftrackStatsBarrelL1 = true;
    if(clusterOfftrackCounterBarrelL2/64 > 0.9)  clusterOfftrackStatsBarrelL2 = true;
    if(clusterOfftrackCounterBarrelL3/88 > 0.9)  clusterOfftrackStatsBarrelL3 = true;
    if(clusterOfftrackCounterEndcapDP1/24 > 0.9) clusterOfftrackStatsEndcapDP1 = true;
    if(clusterOfftrackCounterEndcapDP2/24 > 0.9) clusterOfftrackStatsEndcapDP2 = true;
    if(clusterOfftrackCounterEndcapDM1/24 > 0.9) clusterOfftrackStatsEndcapDM1 = true;
    if(clusterOfftrackCounterEndcapDM2/24 > 0.9) clusterOfftrackStatsEndcapDM2 = true;
  }
  me = bei->get(meName0);
  if(me){
    OffTrackClusterSizeBarrel = bei->get("Pixel/Barrel/BarrelOffTrackClusterSizeCut");
    if(OffTrackClusterSizeBarrel && clusterOfftrackStatsBarrel){
      if(me->hasError()) OffTrackClusterSizeBarrel->Fill(0);
      else OffTrackClusterSizeBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMTRK_size_OffTrack_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_size_OffTrack_Endcap"; 
  me = bei->get(meName0);
  if(me){
    OffTrackClusterSizeEndcap = bei->get("Pixel/Endcap/EndcapOffTrackClusterSizeCut");
    if(OffTrackClusterSizeEndcap && clusterOfftrackStatsEndcap){
      if(me->hasError()) OffTrackClusterSizeEndcap->Fill(0);
      else OffTrackClusterSizeEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(clusterOfftrackStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[5]=0.;
    else if(clusterOfftrackStatsBarrelL1) BarrelL1_cuts_flag_[5]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(clusterOfftrackStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[5]=0.;
    else if(clusterOfftrackStatsBarrelL2) BarrelL2_cuts_flag_[5]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(clusterOfftrackStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[5]=0.;
    else if(clusterOfftrackStatsBarrelL3) BarrelL3_cuts_flag_[5]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(clusterOfftrackStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[5]=0.;
    else if(clusterOfftrackStatsEndcapDM1) EndcapDM1_cuts_flag_[5]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(clusterOfftrackStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[5]=0.;
    else if(clusterOfftrackStatsEndcapDP1) EndcapDP1_cuts_flag_[5]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(clusterOfftrackStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[5]=0.;
    else if(clusterOfftrackStatsEndcapDM2) EndcapDM2_cuts_flag_[5]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(clusterOfftrackStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[5]=0.;
    else if(clusterOfftrackStatsEndcapDP2) EndcapDP2_cuts_flag_[5]=1.;
  }
  if(!Tier0Flag) meName0 = "Pixel/Barrel/SUMTRK_charge_OffTrack_Barrel";
  else meName0 = "Pixel/Barrel/SUMOFF_charge_OffTrack_Barrel"; 
  me = bei->get(meName0);
  if(me){
    OffTrackClusterChargeBarrel = bei->get("Pixel/Barrel/BarrelOffTrackClusterChargeCut");
    if(OffTrackClusterChargeBarrel && clusterOfftrackStatsBarrel){
      if(me->hasError()) OffTrackClusterChargeBarrel->Fill(0);
      else OffTrackClusterChargeBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMTRK_charge_OffTrack_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_charge_OffTrack_Endcap"; 
  me = bei->get(meName0);
  if(me){
    OffTrackClusterChargeEndcap = bei->get("Pixel/Endcap/EndcapOffTrackClusterChargeCut");
    if(OffTrackClusterChargeEndcap && clusterOfftrackStatsEndcap){
      if(me->hasError()) OffTrackClusterChargeEndcap->Fill(0);
      else OffTrackClusterChargeEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_charge_OffTrack_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_charge_OffTrack_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_charge_OffTrack_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_charge_OffTrack_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_charge_OffTrack_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_charge_OffTrack_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_charge_OffTrack_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_charge_OffTrack_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_charge_OffTrack_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_charge_OffTrack_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_charge_OffTrack_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_charge_OffTrack_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_charge_OffTrack_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_charge_OffTrack_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_charge_OffTrack_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_charge_OffTrack_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_charge_OffTrack_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_charge_OffTrack_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_charge_OffTrack_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_charge_OffTrack_Disk_2"; 
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(clusterOfftrackStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[6]=0.;
    else if(clusterOfftrackStatsBarrelL1) BarrelL1_cuts_flag_[6]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(clusterOfftrackStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[6]=0.;
    else if(clusterOfftrackStatsBarrelL2) BarrelL2_cuts_flag_[6]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(clusterOfftrackStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[6]=0.;
    else if(clusterOfftrackStatsBarrelL3) BarrelL3_cuts_flag_[6]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(clusterOfftrackStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[6]=0.;
    else if(clusterOfftrackStatsEndcapDM1) EndcapDM1_cuts_flag_[6]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(clusterOfftrackStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[6]=0.;
    else if(clusterOfftrackStatsEndcapDP1) EndcapDP1_cuts_flag_[6]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(clusterOfftrackStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[6]=0.;
    else if(clusterOfftrackStatsEndcapDM2) EndcapDM2_cuts_flag_[6]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(clusterOfftrackStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[6]=0.;
    else if(clusterOfftrackStatsEndcapDP2) EndcapDP2_cuts_flag_[6]=1.;
  }
  if(!Tier0Flag) meName0 = "Pixel/Barrel/SUMTRK_nclusters_OffTrack_Barrel";
  else meName0 = "Pixel/Barrel/SUMOFF_nclusters_OffTrack_Barrel"; 
  me = bei->get(meName0);
  if(me){
    OffTrackNClustersBarrel = bei->get("Pixel/Barrel/BarrelOffTrackNClustersCut");
    if(OffTrackNClustersBarrel && clusterOfftrackStatsBarrel){
      if(me->hasError()) OffTrackNClustersBarrel->Fill(0);
      else OffTrackNClustersBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMTRK_nclusters_OffTrack_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_nclusters_OffTrack_Endcap"; 
  me = bei->get(meName0);
  if(me){
    OffTrackNClustersEndcap = bei->get("Pixel/Endcap/EndcapOffTrackNClustersCut");
    if(OffTrackNClustersEndcap && clusterOfftrackStatsEndcap){
      if(me->hasError()) OffTrackNClustersEndcap->Fill(0);
      else OffTrackNClustersEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_nclusters_OffTrack_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_nclusters_OffTrack_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_nclusters_OffTrack_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_nclusters_OffTrack_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_nclusters_OffTrack_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_nclusters_OffTrack_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_nclusters_OffTrack_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_nclusters_OffTrack_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_nclusters_OffTrack_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_nclusters_OffTrack_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_nclusters_OffTrack_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_nclusters_OffTrack_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_nclusters_OffTrack_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_nclusters_OffTrack_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_nclusters_OffTrack_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_nclusters_OffTrack_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_nclusters_OffTrack_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_nclusters_OffTrack_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_nclusters_OffTrack_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_nclusters_OffTrack_Disk_2"; 
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(clusterOfftrackStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[7]=0.;
    else if(clusterOfftrackStatsBarrelL1) BarrelL1_cuts_flag_[7]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(clusterOfftrackStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[7]=0.;
    else if(clusterOfftrackStatsBarrelL2) BarrelL2_cuts_flag_[7]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(clusterOfftrackStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[7]=0.;
    else if(clusterOfftrackStatsBarrelL3) BarrelL3_cuts_flag_[7]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(clusterOfftrackStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[7]=0.;
    else if(clusterOfftrackStatsEndcapDM1) EndcapDM1_cuts_flag_[7]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(clusterOfftrackStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[7]=0.;
    else if(clusterOfftrackStatsEndcapDP1) EndcapDP1_cuts_flag_[7]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(clusterOfftrackStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[7]=0.;
    else if(clusterOfftrackStatsEndcapDM2) EndcapDM2_cuts_flag_[7]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(clusterOfftrackStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[7]=0.;
    else if(clusterOfftrackStatsEndcapDP2) EndcapDP2_cuts_flag_[7]=1.;
  }


  // Fill the Rechit flags:
  if(!Tier0Flag){
    meName0 = "Pixel/Barrel/SUMHIT_ErrorX_Barrel";
    if(rechitCounterBarrel/768 > 0.9) rechitStatsBarrel = true;
    if(rechitCounterEndcap/672 > 0.9) rechitStatsEndcap = true;
  }else{
    meName0 = "Pixel/Barrel/SUMOFF_ErrorX_Barrel"; 
    if(rechitCounterBarrel/192 > 0.9) rechitStatsBarrel = true;
    if(rechitCounterEndcap/96 > 0.9) rechitStatsEndcap = true;
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_ErrorX_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_ErrorX_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_ErrorX_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_ErrorX_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_ErrorX_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_ErrorX_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_ErrorX_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_ErrorX_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_ErrorX_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_ErrorX_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_ErrorX_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_ErrorX_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_ErrorX_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_ErrorX_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_ErrorX_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_ErrorX_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_ErrorX_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_ErrorX_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_ErrorX_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_ErrorX_Disk_2"; 
    if(rechitCounterBarrelL1/40 > 0.9)  rechitStatsBarrelL1 = true;
    if(rechitCounterBarrelL2/64 > 0.9)  rechitStatsBarrelL2 = true;
    if(rechitCounterBarrelL3/88 > 0.9)  rechitStatsBarrelL3 = true;
    if(rechitCounterEndcapDP1/24 > 0.9) rechitStatsEndcapDP1 = true;
    if(rechitCounterEndcapDP2/24 > 0.9) rechitStatsEndcapDP2 = true;
    if(rechitCounterEndcapDM1/24 > 0.9) rechitStatsEndcapDM1 = true;
    if(rechitCounterEndcapDM2/24 > 0.9) rechitStatsEndcapDM2 = true;
  }
  me = bei->get(meName0);
  if(me){
    RecHitErrorXBarrel = bei->get("Pixel/Barrel/BarrelRecHitErrorXCut");
    if(RecHitErrorXBarrel && rechitStatsBarrel){
      if(me->hasError()) RecHitErrorXBarrel->Fill(0);
      else RecHitErrorXBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMHIT_ErrorX_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_ErrorX_Endcap"; 
  me = bei->get(meName0);
  if(me){
    RecHitErrorXEndcap = bei->get("Pixel/Endcap/EndcapRecHitErrorXCut");
    if(RecHitErrorXEndcap && rechitStatsEndcap){
      if(me->hasError()) RecHitErrorXEndcap->Fill(0);
      else RecHitErrorXEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(rechitStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[8]=0.;
    else if(rechitStatsBarrelL1) BarrelL1_cuts_flag_[8]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(rechitStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[8]=0.;
    else if(rechitStatsBarrelL2) BarrelL2_cuts_flag_[8]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(rechitStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[8]=0.;
    else if(rechitStatsBarrelL3) BarrelL3_cuts_flag_[8]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(rechitStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[8]=0.;
    else if(rechitStatsEndcapDM1) EndcapDM1_cuts_flag_[8]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(rechitStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[8]=0.;
    else if(rechitStatsEndcapDP1) EndcapDP1_cuts_flag_[8]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(rechitStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[8]=0.;
    else if(rechitStatsEndcapDM2) EndcapDM2_cuts_flag_[8]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(rechitStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[8]=0.;
    else if(rechitStatsEndcapDP2) EndcapDP2_cuts_flag_[8]=1.;
  }
  if(!Tier0Flag) meName0 = "Pixel/Barrel/SUMHIT_ErrorY_Barrel";
  else meName0 = "Pixel/Barrel/SUMOFF_ErrorY_Barrel"; 
  me = bei->get(meName0);
  if(me){
    RecHitErrorYBarrel = bei->get("Pixel/Barrel/BarrelRecHitErrorYCut");
    if(RecHitErrorYBarrel && rechitStatsBarrel){
      if(me->hasError()) RecHitErrorYBarrel->Fill(0);
      else RecHitErrorYBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMHIT_ErrorY_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_ErrorY_Endcap"; 
  me = bei->get(meName0);
  if(me){
    RecHitErrorYEndcap = bei->get("Pixel/Endcap/EndcapRecHitErrorYCut");
    if(RecHitErrorYEndcap && rechitStatsEndcap){
      if(me->hasError()) RecHitErrorYEndcap->Fill(0);
      else RecHitErrorYEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_ErrorY_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_ErrorY_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_ErrorY_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_ErrorY_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_ErrorY_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_ErrorY_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_ErrorY_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_ErrorY_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_ErrorY_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_ErrorY_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_ErrorY_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_ErrorY_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_ErrorY_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_ErrorY_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_ErrorY_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_ErrorY_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_ErrorY_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_ErrorY_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_ErrorY_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_ErrorY_Disk_2"; 
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(rechitStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[9]=0.;
    else if(rechitStatsBarrelL1) BarrelL1_cuts_flag_[9]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(rechitStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[9]=0.;
    else if(rechitStatsBarrelL2) BarrelL2_cuts_flag_[9]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(rechitStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[9]=0.;
    else if(rechitStatsBarrelL3) BarrelL3_cuts_flag_[9]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(rechitStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[9]=0.;
    else if(rechitStatsEndcapDM1) EndcapDM1_cuts_flag_[9]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(rechitStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[9]=0.;
    else if(rechitStatsEndcapDP1) EndcapDP1_cuts_flag_[9]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(rechitStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[9]=0.;
    else if(rechitStatsEndcapDM2) EndcapDM2_cuts_flag_[9]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(rechitStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[9]=0.;
    else if(rechitStatsEndcapDP2) EndcapDP2_cuts_flag_[9]=1.;
  }
 
 
  // Fill the Residual flags:
  if(!Tier0Flag){
    meName0 = "Pixel/Barrel/SUMTRK_residualX_mean_Barrel";
    if(trackCounterBarrel/768 > 0.9) trackStatsBarrel = true;
    if(trackCounterEndcap/672 > 0.9) trackStatsEndcap = true;
  }else{
    meName0 = "Pixel/Barrel/SUMOFF_residualX_mean_Barrel"; 
    if(trackCounterBarrel/192 > 0.9) trackStatsBarrel = true;
    if(trackCounterEndcap/96 > 0.9) trackStatsEndcap = true;
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_residualX_mean_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_residualX_mean_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_residualX_mean_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_residualX_mean_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_residualX_mean_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_residualX_mean_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_residualX_mean_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_residualX_mean_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_residualX_mean_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_residualX_mean_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_residualX_mean_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_residualX_mean_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_residualX_mean_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_residualX_mean_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_residualX_mean_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_residualX_mean_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_residualX_mean_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_residualX_mean_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_residualX_mean_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_residualX_mean_Disk_2"; 
    if(trackCounterBarrelL1/40 > 0.9)  trackStatsBarrelL1 = true;
    if(trackCounterBarrelL2/64 > 0.9)  trackStatsBarrelL2 = true;
    if(trackCounterBarrelL3/88 > 0.9)  trackStatsBarrelL3 = true;
    if(trackCounterEndcapDP1/24 > 0.9) trackStatsEndcapDP1 = true;
    if(trackCounterEndcapDP2/24 > 0.9) trackStatsEndcapDP2 = true;
    if(trackCounterEndcapDM1/24 > 0.9) trackStatsEndcapDM1 = true;
    if(trackCounterEndcapDM2/24 > 0.9) trackStatsEndcapDM2 = true;
  }
  me = bei->get(meName0);
  if(me){
    ResidualXMeanBarrel = bei->get("Pixel/Barrel/BarrelResidualXMeanCut");
    if(ResidualXMeanBarrel && trackStatsBarrel){
      if(me->hasError()) ResidualXMeanBarrel->Fill(0);
      else ResidualXMeanBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMTRK_residualX_mean_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_residualX_mean_Endcap"; 
  me = bei->get(meName0);
  if(me){
    ResidualXMeanEndcap = bei->get("Pixel/Endcap/EndcapResidualXMeanCut");
    if(ResidualXMeanEndcap && trackStatsEndcap){
      if(me->hasError()) ResidualXMeanEndcap->Fill(0);
      else ResidualXMeanEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(trackStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[10]=0.;
    else if(trackStatsBarrelL1) BarrelL1_cuts_flag_[10]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(trackStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[10]=0.;
    else if(trackStatsBarrelL2) BarrelL2_cuts_flag_[10]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(trackStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[10]=0.;
    else if(trackStatsBarrelL3) BarrelL3_cuts_flag_[10]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(trackStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[10]=0.;
    else if(trackStatsEndcapDM1) EndcapDM1_cuts_flag_[10]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(trackStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[10]=0.;
    else if(trackStatsEndcapDP1) EndcapDP1_cuts_flag_[10]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(trackStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[10]=0.;
    else if(trackStatsEndcapDM2) EndcapDM2_cuts_flag_[10]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(trackStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[10]=0.;
    else if(trackStatsEndcapDP2) EndcapDP2_cuts_flag_[10]=1.;
  }
  if(!Tier0Flag) meName0 = "Pixel/Barrel/SUMTRK_residualX_RMS_Barrel";
  else meName0 = "Pixel/Barrel/SUMOFF_residualX_RMS_Barrel"; 
  me = bei->get(meName0);
  if(me){
    ResidualXRMSBarrel = bei->get("Pixel/Barrel/BarrelResidualXRMSCut");
    if(ResidualXRMSBarrel && trackStatsBarrel){
      if(me->hasError()) ResidualXRMSBarrel->Fill(0);
      else ResidualXRMSBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMTRK_residualX_RMS_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_residualX_RMS_Endcap"; 
  me = bei->get(meName0);
  if(me){
    ResidualXRMSEndcap = bei->get("Pixel/Endcap/EndcapResidualXRMSCut");
    if(ResidualXRMSEndcap && trackStatsEndcap){
      if(me->hasError()) ResidualXRMSEndcap->Fill(0);
      else ResidualXRMSEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_residualX_RMS_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_residualX_RMS_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_residualX_RMS_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_residualX_RMS_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_residualX_RMS_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_residualX_RMS_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_residualX_RMS_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_residualX_RMS_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_residualX_RMS_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_residualX_RMS_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_residualX_RMS_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_residualX_RMS_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_residualX_RMS_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_residualX_RMS_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_residualX_RMS_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_residualX_RMS_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_residualX_RMS_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_residualX_RMS_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_residualX_RMS_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_residualX_RMS_Disk_2"; 
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(trackStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[11]=0.;
    else if(trackStatsBarrelL1) BarrelL1_cuts_flag_[11]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(trackStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[11]=0.;
    else if(trackStatsBarrelL2) BarrelL2_cuts_flag_[11]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(trackStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[11]=0.;
    else if(trackStatsBarrelL3) BarrelL3_cuts_flag_[11]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(trackStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[11]=0.;
    else if(trackStatsEndcapDM1) EndcapDM1_cuts_flag_[11]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(trackStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[11]=0.;
    else if(trackStatsEndcapDP1) EndcapDP1_cuts_flag_[11]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(trackStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[11]=0.;
    else if(trackStatsEndcapDM2) EndcapDM2_cuts_flag_[11]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(trackStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[11]=0.;
    else if(trackStatsEndcapDP2) EndcapDP2_cuts_flag_[11]=1.;
  }
  if(!Tier0Flag) meName0 = "Pixel/Barrel/SUMTRK_residualY_mean_Barrel";
  else meName0 = "Pixel/Barrel/SUMOFF_residualY_mean_Barrel"; 
  me = bei->get(meName0);
  if(me){
    ResidualYMeanBarrel = bei->get("Pixel/Barrel/BarrelResidualYMeanCut");
    if(ResidualYMeanBarrel && trackStatsBarrel){
      if(me->hasError()) ResidualYMeanBarrel->Fill(0);
      else ResidualYMeanBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMTRK_residualY_mean_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_residualY_mean_Endcap"; 
  me = bei->get(meName0);
  if(me){
    ResidualYMeanEndcap = bei->get("Pixel/Endcap/EndcapResidualYMeanCut");
    if(ResidualYMeanEndcap && trackStatsEndcap){
      if(me->hasError()) ResidualYMeanEndcap->Fill(0);
      else ResidualYMeanEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_residualY_mean_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_residualY_mean_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_residualY_mean_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_residualY_mean_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_residualY_mean_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_residualY_mean_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_residualY_mean_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_residualY_mean_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_residualY_mean_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_residualY_mean_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_residualY_mean_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_residualY_mean_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_residualY_mean_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_residualY_mean_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_residualY_mean_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_residualY_mean_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_residualY_mean_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_residualY_mean_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_residualY_mean_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_residualY_mean_Disk_2"; 
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(trackStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[12]=0.;
    else if(trackStatsBarrelL1) BarrelL1_cuts_flag_[12]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(trackStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[12]=0.;
    else if(trackStatsBarrelL2) BarrelL2_cuts_flag_[12]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(trackStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[12]=0.;
    else if(trackStatsBarrelL3) BarrelL3_cuts_flag_[12]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(trackStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[12]=0.;
    else if(trackStatsEndcapDM1) EndcapDM1_cuts_flag_[12]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(trackStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[12]=0.;
    else if(trackStatsEndcapDP1) EndcapDP1_cuts_flag_[12]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(trackStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[12]=0.;
    else if(trackStatsEndcapDM2) EndcapDM2_cuts_flag_[12]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(trackStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[12]=0.;
    else if(trackStatsEndcapDP2) EndcapDP2_cuts_flag_[12]=1.;
  }
  if(!Tier0Flag) meName0 = "Pixel/Barrel/SUMTRK_residualY_RMS_Barrel";
  else meName0 = "Pixel/Barrel/SUMOFF_residualY_RMS_Barrel"; 
  me = bei->get(meName0);
  if(me){
    ResidualYRMSBarrel = bei->get("Pixel/Barrel/BarrelResidualYRMSCut");
    if(ResidualYRMSBarrel && trackStatsBarrel){
      if(me->hasError()) ResidualYRMSBarrel->Fill(0);
      else ResidualYRMSBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMTRK_residualY_RMS_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_residualY_RMS_Endcap"; 
  me = bei->get(meName0);
  if(me){
    ResidualYRMSEndcap = bei->get("Pixel/Endcap/EndcapResidualYRMSCut");
    if(ResidualYRMSEndcap && trackStatsEndcap){
      if(me->hasError()) ResidualYRMSEndcap->Fill(0);
      else ResidualYRMSEndcap->Fill(1);
    }
  }
  if(Tier0Flag){
    meName1 = "Pixel/Barrel/Shell_mI/Layer_1/SUMOFF_residualY_RMS_Layer_1"; 
    meName2 = "Pixel/Barrel/Shell_mO/Layer_1/SUMOFF_residualY_RMS_Layer_1"; 
    meName3 = "Pixel/Barrel/Shell_pI/Layer_1/SUMOFF_residualY_RMS_Layer_1"; 
    meName4 = "Pixel/Barrel/Shell_pO/Layer_1/SUMOFF_residualY_RMS_Layer_1"; 
    meName5 = "Pixel/Barrel/Shell_mI/Layer_2/SUMOFF_residualY_RMS_Layer_2"; 
    meName6 = "Pixel/Barrel/Shell_mO/Layer_2/SUMOFF_residualY_RMS_Layer_2"; 
    meName7 = "Pixel/Barrel/Shell_pI/Layer_2/SUMOFF_residualY_RMS_Layer_2"; 
    meName8 = "Pixel/Barrel/Shell_pO/Layer_2/SUMOFF_residualY_RMS_Layer_2"; 
    meName9 = "Pixel/Barrel/Shell_mI/Layer_3/SUMOFF_residualY_RMS_Layer_3"; 
    meName10 = "Pixel/Barrel/Shell_mO/Layer_3/SUMOFF_residualY_RMS_Layer_3"; 
    meName11 = "Pixel/Barrel/Shell_pI/Layer_3/SUMOFF_residualY_RMS_Layer_3"; 
    meName12 = "Pixel/Barrel/Shell_pO/Layer_3/SUMOFF_residualY_RMS_Layer_3"; 
    meName13 = "Pixel/Endcap/HalfCylinder_mI/Disk_1/SUMOFF_residualY_RMS_Disk_1"; 
    meName14 = "Pixel/Endcap/HalfCylinder_mO/Disk_1/SUMOFF_residualY_RMS_Disk_1"; 
    meName15 = "Pixel/Endcap/HalfCylinder_pI/Disk_1/SUMOFF_residualY_RMS_Disk_1"; 
    meName16 = "Pixel/Endcap/HalfCylinder_pO/Disk_1/SUMOFF_residualY_RMS_Disk_1"; 
    meName17 = "Pixel/Endcap/HalfCylinder_mI/Disk_2/SUMOFF_residualY_RMS_Disk_2"; 
    meName18 = "Pixel/Endcap/HalfCylinder_mO/Disk_2/SUMOFF_residualY_RMS_Disk_2"; 
    meName19 = "Pixel/Endcap/HalfCylinder_pI/Disk_2/SUMOFF_residualY_RMS_Disk_2"; 
    meName20 = "Pixel/Endcap/HalfCylinder_pO/Disk_2/SUMOFF_residualY_RMS_Disk_2"; 
    me1 = bei->get(meName1); me2 = bei->get(meName2); me3 = bei->get(meName3); me4 = bei->get(meName4);
    if(trackStatsBarrelL1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL1_cuts_flag_[13]=0.;
    else if(trackStatsBarrelL1) BarrelL1_cuts_flag_[13]=1.;
    me1 = bei->get(meName5); me2 = bei->get(meName6); me3 = bei->get(meName7); me4 = bei->get(meName8);
    if(trackStatsBarrelL2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL2_cuts_flag_[13]=0.;
    else if(trackStatsBarrelL2) BarrelL2_cuts_flag_[13]=1.;
    me1 = bei->get(meName9); me2 = bei->get(meName10); me3 = bei->get(meName11); me4 = bei->get(meName12);
    if(trackStatsBarrelL3 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) BarrelL3_cuts_flag_[13]=0.;
    else if(trackStatsBarrelL3) BarrelL3_cuts_flag_[13]=1.;
    me1 = bei->get(meName13); me2 = bei->get(meName14);
    if(trackStatsEndcapDM1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM1_cuts_flag_[13]=0.;
    else if(trackStatsEndcapDM1) EndcapDM1_cuts_flag_[13]=1.;
    me1 = bei->get(meName15); me2 = bei->get(meName16);
    if(trackStatsEndcapDP1 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP1_cuts_flag_[13]=0.;
    else if(trackStatsEndcapDP1) EndcapDP1_cuts_flag_[13]=1.;
    me1 = bei->get(meName17); me2 = bei->get(meName18);
    if(trackStatsEndcapDM2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDM2_cuts_flag_[13]=0.;
    else if(trackStatsEndcapDM2) EndcapDM2_cuts_flag_[13]=1.;
    me1 = bei->get(meName19); me2 = bei->get(meName20);
    if(trackStatsEndcapDP2 &&
       (( me1 && me1->hasError() ) || 
        ( me2 && me2->hasError() ) ||
        ( me3 && me3->hasError() ) ||
        ( me4 && me4->hasError() ))) EndcapDP2_cuts_flag_[13]=0.;
    else if(trackStatsEndcapDP2) EndcapDP2_cuts_flag_[13]=1.;
  }
  
  // Final combination of all Data Quality results:
  float pixelFlag = -1., barrelFlag = -1., endcapFlag = -1.;
  float f_temp[1]; int i_temp[14]; 
  float combinedCuts = 1.; int numerator = 0, denominator = 0;

  me = bei->get("Pixel/Barrel/BarrelNErrorsCut");
  if(me) f_temp[0] = me->getFloatValue();
  me = bei->get("Pixel/Barrel/BarrelNDigisCut");
  if(me) i_temp[0] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelDigiChargeCut");
  if(me) i_temp[1] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelOnTrackClusterSizeCut");
  if(me) i_temp[2] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelOnTrackNClustersCut");
  if(me) i_temp[3] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelOnTrackClusterChargeCut");
  if(me) i_temp[4] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelOffTrackClusterSizeCut");
  if(me) i_temp[5] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelOffTrackNClustersCut");
  if(me) i_temp[6] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelOffTrackClusterChargeCut");
  if(me) i_temp[7] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelResidualXMeanCut");
  if(me) i_temp[8] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelResidualXRMSCut");
  if(me) i_temp[9] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelResidualYMeanCut");
  if(me) i_temp[10] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelResidualYRMSCut");
  if(me) i_temp[11] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelRecHitErrorXCut");
  if(me) i_temp[12] = me->getIntValue();  
  me = bei->get("Pixel/Barrel/BarrelRecHitErrorYCut");
  if(me) i_temp[13] = me->getIntValue();  
  for(int k=0; k!=14; k++){
    if(i_temp[k]>=0){
      numerator = numerator + i_temp[k];
      denominator++;
      //cout<<"cut flag, Barrel: "<<k<<","<<i_temp[k]<<","<<numerator<<","<<denominator<<endl;
    }
  } 
  if(denominator!=0) combinedCuts = float(numerator)/float(denominator);  
  barrelFlag = f_temp[0] * combinedCuts;
  
  
  //cout<<" the resulting barrel flag is: "<<f_temp[0]<<"*"<<combinedCuts<<"="<<barrelFlag<<endl;
  
  combinedCuts = 1.; numerator = 0; denominator = 0;
  me = bei->get("Pixel/Endcap/EndcapNErrorsCut");
  if(me) f_temp[0] = me->getFloatValue();
  me = bei->get("Pixel/Endcap/EndcapNDigisCut");
  if(me) i_temp[0] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapDigiChargeCut");
  if(me) i_temp[1] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapOnTrackClusterSizeCut");
  if(me) i_temp[2] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapOnTrackNClustersCut");
  if(me) i_temp[3] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapOnTrackClusterChargeCut");
  if(me) i_temp[4] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapOffTrackClusterSizeCut");
  if(me) i_temp[5] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapOffTrackNClustersCut");
  if(me) i_temp[6] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapOffTrackClusterChargeCut");
  if(me) i_temp[7] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapResidualXMeanCut");
  if(me) i_temp[8] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapResidualXRMSCut");
  if(me) i_temp[9] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapResidualYMeanCut");
  if(me) i_temp[10] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapResidualYRMSCut");
  if(me) i_temp[11] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapRecHitErrorXCut");
  if(me) i_temp[12] = me->getIntValue();  
  me = bei->get("Pixel/Endcap/EndcapRecHitErrorYCut");
  if(me) i_temp[13] = me->getIntValue();  
  for(int k=0; k!=14; k++){
    if(i_temp[k]>=0){
      numerator = numerator + i_temp[k];
      denominator++;
      //cout<<"cut flag, Endcap: "<<k<<","<<i_temp[k]<<","<<numerator<<","<<denominator<<endl;
    }
  } 
  if(denominator!=0) combinedCuts = float(numerator)/float(denominator);  
  endcapFlag = f_temp[0] * combinedCuts;
  //cout<<" the resulting endcap flag is: "<<f_temp[0]<<"*"<<combinedCuts<<"="<<endcapFlag<<endl;
  
  pixelFlag = barrelFlag * endcapFlag;
  
  
  
  //cout<<"barrel, endcap, pixel flags: "<<barrelFlag<<","<<endcapFlag<<","<<pixelFlag<<endl;
  SummaryReport = bei->get("Pixel/EventInfo/reportSummary");
  if(SummaryReport) SummaryReport->Fill(pixelFlag);
  SummaryPixel = bei->get("Pixel/EventInfo/reportSummaryContents/PixelDqmFraction");
  if(SummaryPixel) SummaryPixel->Fill(pixelFlag);
  SummaryBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/PixelBarrelDqmFraction");
  if(SummaryBarrel) SummaryBarrel->Fill(barrelFlag);
  SummaryEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/PixelEndcapDqmFraction");
  if(SummaryEndcap)   SummaryEndcap->Fill(endcapFlag);
  }
}

//**********************************************************************************************

void SiPixelDataQuality::fillGlobalQualityPlot(DQMStore * bei, bool init, edm::EventSetup const& eSetup, int nFEDs, bool Tier0Flag)
{
  //calculate eta and phi of the modules and fill a 2D plot:
  if(init){
    allmodsMap = new TH2F("allmodsMap","allmodsMap",40,0.,40.,36,0.,36.);
    errmodsMap = new TH2F("errmodsMap","errmodsMap",40,0.,40.,36,0.,36.);
    goodmodsMap = new TH2F("goodmodsMap","goodmodsMap",40,0.,40.,36,0.,36.);
    count=0; errcount=0;
    //cout<<"Number of FEDs in the readout: "<<nFEDs<<endl;
    SummaryReportMap = bei->get("Pixel/EventInfo/reportSummaryMap");
    if(SummaryReportMap){
      if(!Tier0Flag) for(int i=1; i!=41; i++) for(int j=1; j!=37; j++) SummaryReportMap->setBinContent(i,j,-1.);
      if(Tier0Flag) for(int i=1; i!=8; i++) for(int j=1; j!=16; j++) SummaryReportMap->setBinContent(i,j,-1.);
    }

    init=false;
  }
  
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
// Fill Maps:
  // Online:    
  if(nFEDs==0) return;
  eSetup.get<SiPixelFedCablingMapRcd>().get(theCablingMap);
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  // find a detId for Blades and Barrels (first of the contained Modules!):
  ifstream infile(edm::FileInPath("DQM/SiPixelMonitorClient/test/detId.dat").fullPath().c_str(),ios::in);
  string I_name[1440];
  int I_detId[1440];
  int I_fedId[1440];
  int I_linkId[1440];
  int nModsInFile=0;
  while(!infile.eof()) {
    infile >> I_name[nModsInFile] >> I_detId[nModsInFile] >> I_fedId[nModsInFile] >> I_linkId[nModsInFile] ;
    //cout<<I_name<<" "<<I_detId<<" "<<I_fedId<<" "<<I_linkId ;
    //getline(infile,dummys); //necessary to reach end of record
    infile.close();
    nModsInFile++;
  }
  if(!Tier0Flag && dname.find("Module_")!=string::npos){
    vector<string> meVec = bei->getMEs();
    int detId=-1; int fedId=-1; int linkId=-1;
    for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
      //checking for any digis or FED errors to decide if this module is in DAQ:  
      string full_path = currDir + "/" + (*it);
      //cout<<"path: "<<full_path<<endl;
      if(detId==-1 && full_path.find("SUMOFF")==string::npos &&
         ((full_path.find("ndigis")!=string::npos && full_path.find("SUMDIG")==string::npos) || 
	  (full_path.find("NErrors")!=string::npos && full_path.find("SUMRAW")==string::npos && (getDetId(bei->get(full_path)) > 100)))){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
	if((full_path.find("ndigis")!=string::npos && me->getMean()>0.) ||
	   (full_path.find("NErrors")!=string::npos && me->getMean()>0.)){ 
          detId = getDetId(me);
	  //cout<<"got a module with digis or errors and the detid is: "<<detId<<endl;
          for(int fedid=0; fedid<=40; ++fedid){
            SiPixelFrameConverter converter(theCablingMap.product(),fedid);
	    uint32_t newDetId = detId;
            if(converter.hasDetUnit(newDetId)){
              fedId=fedid;
              break;   
            }
          }
          if(fedId==-1) continue; 
          sipixelobjects::ElectronicIndex cabling; 
          SiPixelFrameConverter formatter(theCablingMap.product(),fedId);
          sipixelobjects::DetectorIndex detector = {detId, 1, 1};      
	  formatter.toCabling(cabling,detector);
          linkId = cabling.link;
	  allmodsMap->Fill(fedId,linkId);
	  //cout<<"this is a module that has digis and/or errors: "<<detId<<","<<fedId<<","<<linkId<<endl;
	  //use presence of any FED error as error flag (except for TBM or ROC resets):
          bool anyerr=false; bool type30=false; bool othererr=false;
          if(full_path.find("ndigis")!=string::npos) full_path = full_path.replace(full_path.find("ndigis"),7,"NErrors");
	  me = bei->get(full_path);
	  if(me) anyerr=true;
          //if(anyerr) cout<<"here is an error: "<<detId<<","<<me->getMean()<<endl;
	  if(full_path.find("NErrors")!=string::npos) full_path = full_path.replace(full_path.find("NErrors"),9,"errorType");
	  me = bei->get(full_path);
	  if(me){
	    for(int jj=1; jj<16; jj++){
	      if(me->getBinContent(jj)>0.){
	        if(jj!=6) othererr=true;
		else type30=true;
	      }
	    }
	    if(type30){
	      full_path = full_path.replace(full_path.find("errorType"),10,"TBMMessage");
	      me = bei->get(full_path);
	      if(me){
	        for(int kk=1; kk<9; kk++){
		  if(me->getBinContent(kk)>0.){
		    if(kk!=6 && kk!=7) othererr=true;
		  }
		}
	      }
	    }
	  }
          if(anyerr && othererr){
	    errmodsMap->Fill(fedId,linkId);
	    //cout<<"this is a module that has errors: "<<detId<<","<<fedId<<","<<linkId<<endl;
	  }
	}
      }
    }
  }
  vector<string> subDirVec = bei->getSubdirs();  
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    bei->cd(*ic);
    init=false;
    fillGlobalQualityPlot(bei,init,eSetup,nFEDs,Tier0Flag);
    bei->goUp();
  }
  
  SummaryReportMap = bei->get("Pixel/EventInfo/reportSummaryMap");
  if(SummaryReportMap){ 
    float contents=0.;
    if(!Tier0Flag){ // Online
      for(int i=1; i!=41; i++)for(int j=1; j!=37; j++){
        //cout<<"bin: "<<i<<","<<j<<endl;
        contents = (allmodsMap->GetBinContent(i,j))-(errmodsMap->GetBinContent(i,j));
        goodmodsMap->SetBinContent(i,j,contents);
        //cout<<"\t the map: "<<allmodsMap->GetBinContent(i,j)<<","<<errmodsMap->GetBinContent(i,j)<<endl;
        if(allmodsMap->GetBinContent(i,j)>0){
          contents = (goodmodsMap->GetBinContent(i,j))/(allmodsMap->GetBinContent(i,j));
        }else{
          contents = -1.;
        }
        //cout<<"\t\t MAP: "<<i<<","<<j<<","<<contents<<endl;
        SummaryReportMap->setBinContent(i,j,contents);
      }
    }else{ // Offline
      SummaryReportMap->setBinContent(1,1,BarrelL1_error_flag_);
      SummaryReportMap->setBinContent(2,1,BarrelL2_error_flag_);
      SummaryReportMap->setBinContent(3,1,BarrelL3_error_flag_);
      SummaryReportMap->setBinContent(4,1,EndcapDM2_error_flag_);
      SummaryReportMap->setBinContent(5,1,EndcapDM1_error_flag_);
      SummaryReportMap->setBinContent(6,1,EndcapDP1_error_flag_);
      SummaryReportMap->setBinContent(7,1,EndcapDP2_error_flag_);
      for(int j=2; j!=16; j++){
        SummaryReportMap->setBinContent(1,j,BarrelL1_cuts_flag_[j-2]);
        SummaryReportMap->setBinContent(2,j,BarrelL2_cuts_flag_[j-2]);
        SummaryReportMap->setBinContent(3,j,BarrelL3_cuts_flag_[j-2]);
        SummaryReportMap->setBinContent(4,j,EndcapDM2_cuts_flag_[j-2]);
        SummaryReportMap->setBinContent(5,j,EndcapDM1_cuts_flag_[j-2]);
        SummaryReportMap->setBinContent(6,j,EndcapDP1_cuts_flag_[j-2]);
        SummaryReportMap->setBinContent(7,j,EndcapDP2_cuts_flag_[j-2]);
      }
    }
  }
  if(allmodsMap) allmodsMap->Clear();
  if(goodmodsMap) goodmodsMap->Clear();
  if(errmodsMap) errmodsMap->Clear();


  //cout<<"counters: "<<count<<" , "<<errcount<<endl;
}

//**********************************************************************************************

