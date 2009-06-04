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

#include <qstring.h>
#include <qregexp.h>

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
 QRegExp rx("(\\w+)_(\\w+)_(\\d+)") ;
 QString mEName = mE->getName() ;

 int detId = 0;
 
 if( rx.search(mEName) != -1 )
 {
  detId = rx.cap(3).toInt() ;
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
  SummaryReportMap = bei->book2D("reportSummaryMap","Pixel Summary Map",7,0.,7.,24,1.,25.);
  SummaryReportMap->setBinLabel(1,"Barrel_Layer_1",1);
  SummaryReportMap->setBinLabel(2,"Barrel_Layer_2",1);
  SummaryReportMap->setBinLabel(3,"Barrel_Layer_3",1);
  SummaryReportMap->setBinLabel(4,"Endcap_Disk_1 -z",1);
  SummaryReportMap->setBinLabel(5,"Endcap_Disk_2 -z",1);
  SummaryReportMap->setBinLabel(6,"Endcap_Disk_1 +z",1);
  SummaryReportMap->setBinLabel(7,"Endcap_Disk_2 +z",1);
    
  bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents");
    SummaryPixel = bei->bookFloat("PixelDqmFraction");
    SummaryBarrel = bei->bookFloat("PixelBarrelDqmFraction");
    SummaryEndcap = bei->bookFloat("PixelEndcapDqmFraction");
  // book the data certification cuts:
    NErrorsBarrel = bei->bookFloat("BarrelNErrorsCut");
    NErrorsEndcap = bei->bookFloat("EndcapNErrorsCut");
    NErrorsFEDs = bei->bookFloat("FEDsNErrorsCut");
    NDigisBarrel = bei->bookInt("BarrelNDigisCut");
    NDigisEndcap = bei->bookInt("EndcapNDigisCut");
    DigiChargeBarrel = bei->bookInt("BarrelDigiChargeCut");
    DigiChargeEndcap = bei->bookInt("EndcapDigiChargeCut");
    OnTrackClusterSizeBarrel = bei->bookInt("BarrelOnTrackClusterSizeCut");
    OnTrackClusterSizeEndcap = bei->bookInt("EndcapOnTrackClusterSizeCut");
    OnTrackNClustersBarrel = bei->bookInt("BarrelOnTrackNClustersCut");
    OnTrackNClustersEndcap = bei->bookInt("EndcapOnTrackNClustersCut");
    OnTrackClusterChargeBarrel = bei->bookInt("BarrelOnTrackClusterChargeCut");
    OnTrackClusterChargeEndcap = bei->bookInt("EndcapOnTrackClusterChargeCut");
    OffTrackClusterSizeBarrel = bei->bookInt("BarrelOffTrackClusterSizeCut");
    OffTrackClusterSizeEndcap = bei->bookInt("EndcapOffTrackClusterSizeCut");
    OffTrackNClustersBarrel = bei->bookInt("BarrelOffTrackNClustersCut");
    OffTrackNClustersEndcap = bei->bookInt("EndcapOffTrackNClustersCut");
    OffTrackClusterChargeBarrel = bei->bookInt("BarrelOffTrackClusterChargeCut");
    OffTrackClusterChargeEndcap = bei->bookInt("EndcapOffTrackClusterChargeCut");
    ResidualXMeanBarrel = bei->bookInt("BarrelResidualXMeanCut");
    ResidualXMeanEndcap = bei->bookInt("EndcapResidualXMeanCut");
    ResidualXRMSBarrel = bei->bookInt("BarrelResidualXRMSCut");
    ResidualXRMSEndcap = bei->bookInt("EndcapResidualXRMSCut");
    ResidualYMeanBarrel = bei->bookInt("BarrelResidualYMeanCut");
    ResidualYMeanEndcap = bei->bookInt("EndcapResidualYMeanCut");
    ResidualYRMSBarrel = bei->bookInt("BarrelResidualYRMSCut");
    ResidualYRMSEndcap = bei->bookInt("EndcapResidualYRMSCut");
    RecHitErrorXBarrel = bei->bookInt("BarrelRecHitErrorXCut");
    RecHitErrorYBarrel = bei->bookInt("BarrelRecHitErrorYCut");
    RecHitErrorXEndcap = bei->bookInt("EndcapRecHitErrorXCut");
    RecHitErrorYEndcap = bei->bookInt("EndcapRecHitErrorYCut");
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
    
    //MonitoringElements:
    SummaryReport = bei->get("Pixel/EventInfo/reportSummary");
    if(SummaryReport) SummaryReport->Fill(-1.);
    SummaryPixel = bei->get("Pixel/EventInfo/reportSummaryContents/PixelDqmFraction");
    if(SummaryPixel) SummaryPixel->Fill(-1.);
    SummaryBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/PixelBarrelDqmFraction");
    if(SummaryBarrel) SummaryBarrel->Fill(-1.);
    SummaryEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/PixelEndcapDqmFraction");
    if(SummaryEndcap)	SummaryEndcap->Fill(-1.);
    NErrorsBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelNErrorsCut");
    if(NErrorsBarrel) NErrorsBarrel->Fill(-1.);
    NErrorsEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapNErrorsCut");
    if(NErrorsEndcap) NErrorsEndcap->Fill(-1.);
    NErrorsFEDs = bei->get("Pixel/EventInfo/reportSummaryContents/FEDsNErrorsCut");
    if(NErrorsFEDs) NErrorsFEDs->Fill(-1.);
    NDigisBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelNDigisCut");
    if(NDigisBarrel) NDigisBarrel->Fill(-1);
    NDigisEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapNDigisCut");
    if(NDigisEndcap) NDigisEndcap->Fill(-1);
    DigiChargeBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelDigiChargeCut");
    if(DigiChargeBarrel) DigiChargeBarrel->Fill(-1);
    DigiChargeEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapDigiChargeCut");
    if(DigiChargeEndcap) DigiChargeEndcap->Fill(-1);
    OnTrackClusterSizeBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOnTrackClusterSizeCut");
    if(OnTrackClusterSizeBarrel) OnTrackClusterSizeBarrel->Fill(-1);
    OnTrackClusterSizeEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOnTrackClusterSizeCut");
    if(OnTrackClusterSizeEndcap) OnTrackClusterSizeEndcap->Fill(-1);
    OnTrackClusterChargeBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOnTrackClusterChargeCut");
    if(OnTrackClusterChargeBarrel) OnTrackClusterChargeBarrel->Fill(-1);
    OnTrackClusterChargeEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOnTrackClusterChargeCut");
    if(OnTrackClusterChargeEndcap) OnTrackClusterChargeEndcap->Fill(-1);
    OnTrackNClustersBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOnTrackNClustersCut");
    if(OnTrackNClustersBarrel) OnTrackNClustersBarrel->Fill(-1);
    OnTrackNClustersEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOnTrackNClustersCut");
    if(OnTrackNClustersEndcap) OnTrackNClustersEndcap->Fill(-1);
    OffTrackClusterSizeBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOffTrackClusterSizeCut");
    if(OffTrackClusterSizeBarrel) OffTrackClusterSizeBarrel->Fill(-1);
    OffTrackClusterSizeEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOffTrackClusterSizeCut");
    if(OffTrackClusterSizeEndcap) OffTrackClusterSizeEndcap->Fill(-1);
    OffTrackClusterChargeBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOffTrackClusterChargeCut");
    if(OffTrackClusterChargeBarrel) OffTrackClusterChargeBarrel->Fill(-1);
    OffTrackClusterChargeEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOffTrackClusterChargeCut");
    if(OffTrackClusterChargeEndcap) OffTrackClusterChargeEndcap->Fill(-1);
    OffTrackNClustersBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOffTrackNClustersCut");
    if(OffTrackNClustersBarrel) OffTrackNClustersBarrel->Fill(-1);
    OffTrackNClustersEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOffTrackNClustersCut");
    if(OffTrackNClustersEndcap) OffTrackNClustersEndcap->Fill(-1);
    ResidualXMeanBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelResidualXMeanCut");
    if(ResidualXMeanBarrel) ResidualXMeanBarrel->Fill(-1);
    ResidualXMeanEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapResidualXMeanCut");
    if(ResidualXMeanEndcap) ResidualXMeanEndcap->Fill(-1);
    ResidualXRMSBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelResidualXRMSCut");
    if(ResidualXRMSBarrel) ResidualXRMSBarrel->Fill(-1);
    ResidualXRMSEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapResidualXRMSCut");
    if(ResidualXRMSEndcap) ResidualXRMSEndcap->Fill(-1);
    ResidualYMeanBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelResidualYMeanCut");
    if(ResidualYMeanBarrel) ResidualYMeanBarrel->Fill(-1);
    ResidualYMeanEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapResidualYMeanCut");
    if(ResidualYMeanEndcap) ResidualYMeanEndcap->Fill(-1);
    ResidualYRMSBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelResidualYRMSCut");
    if(ResidualYRMSBarrel) ResidualYRMSBarrel->Fill(-1);
    ResidualYRMSEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapResidualYRMSCut");
    if(ResidualYRMSEndcap) ResidualYRMSEndcap->Fill(-1);
    RecHitErrorXBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelRecHitErrorXCut");
    if(RecHitErrorXBarrel) RecHitErrorXBarrel->Fill(-1);
    RecHitErrorYBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelRecHitErrorYCut");
    if(RecHitErrorYBarrel) RecHitErrorYBarrel->Fill(-1);
    RecHitErrorXEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapRecHitErrorXCut");
    if(RecHitErrorXEndcap) RecHitErrorXEndcap->Fill(-1);
    RecHitErrorYEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapRecHitErrorYCut");
    if(RecHitErrorYEndcap) RecHitErrorYEndcap->Fill(-1);
    init=false;
  }
  if(nFEDs==0) return;  
  
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);

  QRegExp rx, rxb, rxe;
  if(!Tier0Flag) rx = QRegExp("Module_");
  else { rxb = QRegExp("Ladder_"); rxe = QRegExp("Blade_"); }
 
  bool digiStatsBarrel = false, clusterOntrackStatsBarrel = false, clusterOfftrackStatsBarrel = false, rechitStatsBarrel = false, trackStatsBarrel = false;
  int digiCounterBarrel = 0, clusterOntrackCounterBarrel = 0, clusterOfftrackCounterBarrel = 0, rechitCounterBarrel = 0, trackCounterBarrel = 0;
  bool digiStatsEndcap = false, clusterOntrackStatsEndcap = false, clusterOfftrackStatsEndcap = false, rechitStatsEndcap = false, trackStatsEndcap = false;
  int digiCounterEndcap = 0, clusterOntrackCounterEndcap = 0, clusterOfftrackCounterEndcap = 0, rechitCounterEndcap = 0, trackCounterEndcap = 0;
  
  if((!Tier0Flag && rx.search(dname)!=-1) || 
     (Tier0Flag && (rxb.search(dname)!=-1 || rxe.search(dname)!=-1))){

    objectCount_++;

    if(currDir.find("Pixel")!=string::npos) allMods_++;
    if(currDir.find("Barrel")!=string::npos) barrelMods_++;
    if(currDir.find("Endcap")!=string::npos) endcapMods_++;
      
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
	  }
        }	
      }else if(full_path.find("_ndigis_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>50){
	  if(full_path.find("Barrel")!=string::npos) digiCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) digiCounterEndcap++;
        }
      }else if(full_path.find("_nclusters_OnTrack_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>50){
	  if(full_path.find("Barrel")!=string::npos) clusterOntrackCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) clusterOntrackCounterEndcap++;
        }
      }else if(full_path.find("_nclusters_OffTrack_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>50){
	  if(full_path.find("Barrel")!=string::npos) clusterOfftrackCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) clusterOfftrackCounterEndcap++;
        }
      }else if(full_path.find("_nRecHits_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>50){
	  if(full_path.find("Barrel")!=string::npos) rechitCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) rechitCounterEndcap++;
        }
      }else if(full_path.find("_residualX_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>50){
	  if(full_path.find("Barrel")!=string::npos) trackCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) trackCounterEndcap++;
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
  NErrorsBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelNErrorsCut");
  if(NErrorsBarrel) NErrorsBarrel->Fill(barrel_error_flag_);
  NErrorsEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapNErrorsCut");
  if(NErrorsEndcap)   NErrorsEndcap->Fill(endcap_error_flag_);
  NErrorsFEDs = bei->get("Pixel/EventInfo/reportSummaryContents/FEDsNErrorsCut");
  if(NErrorsFEDs) NErrorsFEDs->Fill(1.); // hardwired for the moment, need to fix!

  
  string meName; MonitorElement * me;
  
  // Fill the Digi flags:
  if(!Tier0Flag){
    meName = "Pixel/Barrel/SUMDIG_ndigis_Barrel";
    if(digiCounterBarrel/768 > 0.9) digiStatsBarrel = true;
    if(digiCounterEndcap/672 > 0.9) digiStatsEndcap = true;
  }else{
    meName = "Pixel/Barrel/SUMOFF_ndigis_Barrel"; 
    if(digiCounterBarrel/192 > 0.9) digiStatsBarrel = true;
    if(digiCounterEndcap/96 > 0.9) digiStatsEndcap = true;
  }
  me = bei->get(meName);
  if(me){
    NDigisBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelNDigisCut");
    if(NDigisBarrel){
      if(me->hasError() && digiStatsBarrel) NDigisBarrel->Fill(0);
      else NDigisBarrel->Fill(1); 
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMDIG_ndigis_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_ndigis_Endcap"; 
  me = bei->get(meName);
  if(me){
    NDigisEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapNDigisCut");
    if(NDigisEndcap){
      if(me->hasError() && digiStatsEndcap) NDigisEndcap->Fill(0);
      else NDigisEndcap->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Barrel/SUMDIG_adc_Barrel";
  else meName = "Pixel/Barrel/SUMOFF_adc_Barrel"; 
  me = bei->get(meName);
  if(me){
    DigiChargeBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelDigiChargeCut");
    if(DigiChargeBarrel){
      if(me->hasError() && digiStatsBarrel) DigiChargeBarrel->Fill(0);
      else DigiChargeBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMDIG_adc_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_adc_Endcap"; 
  me = bei->get(meName);
  if(me){
    DigiChargeEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapDigiChargeCut");
    if(DigiChargeEndcap){
      if(me->hasError() && digiStatsEndcap) DigiChargeEndcap->Fill(0);
      else DigiChargeEndcap->Fill(1);
    }
  }
     
     
    // Fill the OnTrack Cluster flags:
  if(!Tier0Flag){
    meName = "Pixel/Barrel/SUMTRK_size_OnTrack_Barrel";
    if(clusterOntrackCounterBarrel/768 > 0.9) clusterOntrackStatsBarrel = true;
    if(clusterOntrackCounterEndcap/672 > 0.9) clusterOntrackStatsEndcap = true;
  }else{  
    meName = "Pixel/Barrel/SUMOFF_size_OnTrack_Barrel"; 
    if(clusterOntrackCounterBarrel/192 > 0.9) clusterOntrackStatsBarrel = true;
    if(clusterOntrackCounterEndcap/96 > 0.9) clusterOntrackStatsEndcap = true;
  }
  me = bei->get(meName);
  if(me){
    OnTrackClusterSizeBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOnTrackClusterSizeCut");
    if(OnTrackClusterSizeBarrel){
      if(me->hasError() && clusterOntrackStatsBarrel) OnTrackClusterSizeBarrel->Fill(0);
      else OnTrackClusterSizeBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMTRK_size_OnTrack_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_size_OnTrack_Endcap"; 
  me = bei->get(meName);
  if(me){
    OnTrackClusterSizeEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOnTrackClusterSizeCut");
    if(OnTrackClusterSizeEndcap){
      if(me->hasError() && clusterOntrackStatsEndcap) OnTrackClusterSizeEndcap->Fill(0);
      else OnTrackClusterSizeEndcap->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Barrel/SUMTRK_charge_OnTrack_Barrel";
  else meName = "Pixel/Barrel/SUMOFF_charge_OnTrack_Barrel"; 
  me = bei->get(meName);
  if(me){
    OnTrackClusterChargeBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOnTrackClusterChargeCut");
    if(OnTrackClusterChargeBarrel){
      if(me->hasError() && clusterOntrackStatsBarrel) OnTrackClusterChargeBarrel->Fill(0);
      else OnTrackClusterChargeBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMTRK_charge_OnTrack_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_charge_OnTrack_Endcap"; 
  me = bei->get(meName);
  if(me){
    OnTrackClusterChargeEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOnTrackClusterChargeCut");
    if(OnTrackClusterChargeEndcap){
      if(me->hasError() && clusterOntrackStatsEndcap) OnTrackClusterChargeEndcap->Fill(0);
      else OnTrackClusterChargeEndcap->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Barrel/SUMTRK_nclusters_OnTrack_Barrel";
  else meName = "Pixel/Barrel/SUMOFF_nclusters_OnTrack_Barrel"; 
  me = bei->get(meName);
  if(me){
    OnTrackNClustersBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOnTrackNClustersCut");
    if(OnTrackNClustersBarrel){
      if(me->hasError() && clusterOntrackStatsBarrel) OnTrackNClustersBarrel->Fill(0);
      else OnTrackNClustersBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMTRK_nclusters_OnTrack_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_nclusters_OnTrack_Endcap"; 
  me = bei->get(meName);
  if(me){
    OnTrackNClustersEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOnTrackNClustersCut");
    if(OnTrackNClustersEndcap){
      if(me->hasError() && clusterOntrackStatsEndcap) OnTrackNClustersEndcap->Fill(0);
      else OnTrackNClustersEndcap->Fill(1);
    }
  }

     
  // Fill the OffTrack Cluster flags:
  if(!Tier0Flag){
    meName = "Pixel/Barrel/SUMTRK_size_OffTrack_Barrel";
    if(clusterOfftrackCounterBarrel/768 > 0.9) clusterOfftrackStatsBarrel = true;
    if(clusterOfftrackCounterEndcap/672 > 0.9) clusterOfftrackStatsEndcap = true;
  }else{
    meName = "Pixel/Barrel/SUMOFF_size_OffTrack_Barrel"; 
    if(clusterOfftrackCounterBarrel/192 > 0.9) clusterOfftrackStatsBarrel = true;
    if(clusterOfftrackCounterEndcap/96 > 0.9) clusterOfftrackStatsEndcap = true;
  }
  me = bei->get(meName);
  if(me){
    OffTrackClusterSizeBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOffTrackClusterSizeCut");
    if(OffTrackClusterSizeBarrel){
      if(me->hasError() && clusterOfftrackStatsBarrel) OffTrackClusterSizeBarrel->Fill(0);
      else OffTrackClusterSizeBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMTRK_size_OffTrack_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_size_OffTrack_Endcap"; 
  me = bei->get(meName);
  if(me){
    OffTrackClusterSizeEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOffTrackClusterSizeCut");
    if(OffTrackClusterSizeEndcap){
      if(me->hasError() && clusterOfftrackStatsEndcap) OffTrackClusterSizeEndcap->Fill(0);
      else OffTrackClusterSizeEndcap->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Barrel/SUMTRK_charge_OffTrack_Barrel";
  else meName = "Pixel/Barrel/SUMOFF_charge_OffTrack_Barrel"; 
  me = bei->get(meName);
  if(me){
    OffTrackClusterChargeBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOffTrackClusterChargeCut");
    if(OffTrackClusterChargeBarrel){
      if(me->hasError() && clusterOfftrackStatsBarrel) OffTrackClusterChargeBarrel->Fill(0);
      else OffTrackClusterChargeBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMTRK_charge_OffTrack_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_charge_OffTrack_Endcap"; 
  me = bei->get(meName);
  if(me){
    OffTrackClusterChargeEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOffTrackClusterChargeCut");
    if(OffTrackClusterChargeEndcap){
      if(me->hasError() && clusterOfftrackStatsEndcap) OffTrackClusterChargeEndcap->Fill(0);
      else OffTrackClusterChargeEndcap->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Barrel/SUMTRK_nclusters_OffTrack_Barrel";
  else meName = "Pixel/Barrel/SUMOFF_nclusters_OffTrack_Barrel"; 
  me = bei->get(meName);
  if(me){
    OffTrackNClustersBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOffTrackNClustersCut");
    if(OffTrackNClustersBarrel){
      if(me->hasError() && clusterOfftrackStatsBarrel) OffTrackNClustersBarrel->Fill(0);
      else OffTrackNClustersBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMTRK_nclusters_OffTrack_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_nclusters_OffTrack_Endcap"; 
  me = bei->get(meName);
  if(me){
    OffTrackNClustersEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOffTrackNClustersCut");
    if(OffTrackNClustersEndcap){
      if(me->hasError() && clusterOfftrackStatsEndcap) OffTrackNClustersEndcap->Fill(0);
      else OffTrackNClustersEndcap->Fill(1);
    }
  }


  // Fill the Rechit flags:
  if(!Tier0Flag){
    meName = "Pixel/Barrel/SUMHIT_ErrorX_Barrel";
    if(rechitCounterBarrel/768 > 0.9) rechitStatsBarrel = true;
    if(rechitCounterEndcap/672 > 0.9) rechitStatsEndcap = true;
  }else{
    meName = "Pixel/Barrel/SUMOFF_ErrorX_Barrel"; 
    if(rechitCounterBarrel/192 > 0.9) rechitStatsBarrel = true;
    if(rechitCounterEndcap/96 > 0.9) rechitStatsEndcap = true;
  }
  me = bei->get(meName);
  if(me){
    RecHitErrorXBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelRecHitErrorXCut");
    if(RecHitErrorXBarrel){
      if(me->hasError() && rechitStatsBarrel) RecHitErrorXBarrel->Fill(0);
      else RecHitErrorXBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMHIT_ErrorX_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_ErrorX_Endcap"; 
  me = bei->get(meName);
  if(me){
    RecHitErrorXEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapRecHitErrorXCut");
    if(RecHitErrorXEndcap){
      if(me->hasError() && rechitStatsEndcap) RecHitErrorXEndcap->Fill(0);
      else RecHitErrorXEndcap->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Barrel/SUMHIT_ErrorY_Barrel";
  else meName = "Pixel/Barrel/SUMOFF_ErrorY_Barrel"; 
  me = bei->get(meName);
  if(me){
    RecHitErrorYBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelRecHitErrorYCut");
    if(RecHitErrorYBarrel){
      if(me->hasError() && rechitStatsBarrel) RecHitErrorYBarrel->Fill(0);
      else RecHitErrorYBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMHIT_ErrorY_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_ErrorY_Endcap"; 
  me = bei->get(meName);
  if(me){
    RecHitErrorYEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapRecHitErrorYCut");
    if(RecHitErrorYEndcap){
      if(me->hasError() && rechitStatsEndcap) RecHitErrorYEndcap->Fill(0);
      else RecHitErrorYEndcap->Fill(1);
    }
  }
 
 
  // Fill the Residual flags:
  if(!Tier0Flag){
    meName = "Pixel/Barrel/SUMTRK_residualX_mean_Barrel";
    if(trackCounterBarrel/768 > 0.9) trackStatsBarrel = true;
    if(trackCounterEndcap/672 > 0.9) trackStatsEndcap = true;
  }else{
    meName = "Pixel/Barrel/SUMOFF_residualX_mean_Barrel"; 
    if(trackCounterBarrel/192 > 0.9) trackStatsBarrel = true;
    if(trackCounterEndcap/96 > 0.9) trackStatsEndcap = true;
  }
  me = bei->get(meName);
  if(me){
    ResidualXMeanBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelResidualXMeanCut");
    if(ResidualXMeanBarrel){
      if(me->hasError() && trackStatsBarrel) ResidualXMeanBarrel->Fill(0);
      else ResidualXMeanBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMTRK_residualX_mean_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_residualX_mean_Endcap"; 
  me = bei->get(meName);
  if(me){
    ResidualXMeanEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapResidualXMeanCut");
    if(ResidualXMeanEndcap){
      if(me->hasError() && trackStatsEndcap) ResidualXMeanEndcap->Fill(0);
      else ResidualXMeanEndcap->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Barrel/SUMTRK_residualX_RMS_Barrel";
  else meName = "Pixel/Barrel/SUMOFF_residualX_RMS_Barrel"; 
  me = bei->get(meName);
  if(me){
    ResidualXRMSBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelResidualXRMSCut");
    if(ResidualXRMSBarrel){
      if(me->hasError() && trackStatsBarrel) ResidualXRMSBarrel->Fill(0);
      else ResidualXRMSBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMTRK_residualX_RMS_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_residualX_RMS_Endcap"; 
  me = bei->get(meName);
  if(me){
    ResidualXRMSEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapResidualXRMSCut");
    if(ResidualXRMSEndcap){
      if(me->hasError() && trackStatsEndcap) ResidualXRMSEndcap->Fill(0);
      else ResidualXRMSEndcap->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Barrel/SUMTRK_residualY_mean_Barrel";
  else meName = "Pixel/Barrel/SUMOFF_residualY_mean_Barrel"; 
  me = bei->get(meName);
  if(me){
    ResidualYMeanBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelResidualYMeanCut");
    if(ResidualYMeanBarrel){
      if(me->hasError() && trackStatsBarrel) ResidualYMeanBarrel->Fill(0);
      else ResidualYMeanBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMTRK_residualY_mean_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_residualY_mean_Endcap"; 
  me = bei->get(meName);
  if(me){
    ResidualYMeanEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapResidualYMeanCut");
    if(ResidualYMeanEndcap){
      if(me->hasError() && trackStatsEndcap) ResidualYMeanEndcap->Fill(0);
      else ResidualYMeanEndcap->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Barrel/SUMTRK_residualY_RMS_Barrel";
  else meName = "Pixel/Barrel/SUMOFF_residualY_RMS_Barrel"; 
  me = bei->get(meName);
  if(me){
    ResidualYRMSBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelResidualYRMSCut");
    if(ResidualYRMSBarrel){
      if(me->hasError() && trackStatsBarrel) ResidualYRMSBarrel->Fill(0);
      else ResidualYRMSBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName = "Pixel/Endcap/SUMTRK_residualY_RMS_Endcap";
  else meName = "Pixel/Endcap/SUMOFF_residualY_RMS_Endcap"; 
  me = bei->get(meName);
  if(me){
    ResidualYRMSEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapResidualYRMSCut");
    if(ResidualYRMSEndcap){
      if(me->hasError() && trackStatsEndcap) ResidualYRMSEndcap->Fill(0);
      else ResidualYRMSEndcap->Fill(1);
    }
  }
  
  // Final combination of all Data Quality results:
  float pixelFlag = -1., barrelFlag = -1., endcapFlag = -1.;
  float f_temp[1]; int i_temp[14]; 
  float combinedCuts = 1.; int numerator = 0, denominator = 0;

  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelNErrorsCut");
  if(me) f_temp[0] = me->getFloatValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelNDigisCut");
  if(me) i_temp[0] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelDigiChargeCut");
  if(me) i_temp[1] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOnTrackClusterSizeCut");
  if(me) i_temp[2] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOnTrackNClustersCut");
  if(me) i_temp[3] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOnTrackClusterChargeCut");
  if(me) i_temp[4] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOffTrackClusterSizeCut");
  if(me) i_temp[5] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOffTrackNClustersCut");
  if(me) i_temp[6] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelOffTrackClusterChargeCut");
  if(me) i_temp[7] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelResidualXMeanCut");
  if(me) i_temp[8] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelResidualXRMSCut");
  if(me) i_temp[9] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelResidualYMeanCut");
  if(me) i_temp[10] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelResidualYRMSCut");
  if(me) i_temp[11] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelRecHitErrorXCut");
  if(me) i_temp[12] = me->getIntValue();  
  me = bei->get("Pixel/EventInfo/reportSummaryContents/BarrelRecHitErrorYCut");
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
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapNErrorsCut");
  if(me) f_temp[0] = me->getFloatValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapNDigisCut");
  if(me) i_temp[0] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapDigiChargeCut");
  if(me) i_temp[1] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOnTrackClusterSizeCut");
  if(me) i_temp[2] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOnTrackNClustersCut");
  if(me) i_temp[3] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOnTrackClusterChargeCut");
  if(me) i_temp[4] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOffTrackClusterSizeCut");
  if(me) i_temp[5] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOffTrackNClustersCut");
  if(me) i_temp[6] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapOffTrackClusterChargeCut");
  if(me) i_temp[7] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapResidualXMeanCut");
  if(me) i_temp[8] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapResidualXRMSCut");
  if(me) i_temp[9] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapResidualYMeanCut");
  if(me) i_temp[10] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapResidualYRMSCut");
  if(me) i_temp[11] = me->getIntValue();
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapRecHitErrorXCut");
  if(me) i_temp[12] = me->getIntValue();  
  me = bei->get("Pixel/EventInfo/reportSummaryContents/EndcapRecHitErrorYCut");
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
    if(SummaryReportMap) for(int i=1; i!=8; i++) for(int j=1; j!=2; j++) SummaryReportMap->setBinContent(i,j,-1.);
    NErrorsFEDMap = bei->get("Pixel/EventInfo/reportSummaryContents/FEDErrorsCut/NErrorsFEDMap");
    if(NErrorsFEDMap) for(int i=1; i!=41; i++) for(int j=1; j!=37; j++) NErrorsFEDMap->setBinContent(i,j,-1.);
    NDigisMap = bei->get("Pixel/EventInfo/reportSummaryContents/NDigisCut/NDigisMap");
    if(NDigisMap) for(int i=1; i!=8; i++) for(int j=1; j!=2; j++) NDigisMap->setBinContent(i,j,-1.);

    init=false;
  }
  
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
// Fill Maps:
  if(nFEDs==0) return;
  eSetup.get<SiPixelFedCablingMapRcd>().get(theCablingMap);
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  QRegExp rx, rxb, rxe;
  if(!Tier0Flag) rx = QRegExp("Module_");
  else { rxb = QRegExp("Ladder_"); rxe = QRegExp("Blade_"); }
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
  if((!Tier0Flag && rx.search(dname)!=-1) || 
     (Tier0Flag && (rxb.search(dname)!=-1 || rxe.search(dname)!=-1))){
    vector<string> meVec = bei->getMEs();
    int detId=-1; int fedId=-1; int linkId=-1;
    for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
      //checking for any digis or FED errors to decide if this module is in DAQ:  
      string full_path = currDir + "/" + (*it);
      //cout<<"path: "<<full_path<<endl;
//+++++++++++++ Fill NErrorsFEDMap:      
      if(!Tier0Flag && detId==-1 && full_path.find("SUMOFF")==string::npos &&
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
      }else if(Tier0Flag && detId==-1 && full_path.find("SUMOFF")==string::npos && 
               ((full_path.find("ndigis")!=string::npos && full_path.find("SUMDIG")==string::npos) || 
	        (full_path.find("NErrors")!=string::npos && full_path.find("SUMRAW")==string::npos && (getDetId(bei->get(full_path)) > 100)))){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
	if((full_path.find("ndigis")!=string::npos && me->getMean()>0.) ||
	   (full_path.find("NErrors")!=string::npos && me->getMean()>0.)){ 
	  string only_path = full_path.substr(0,full_path.size()-full_path.substr(full_path.find_last_of("/")+1).size()-1);
	  int thisFedId[4] = {-1,-1,-1,-1}, thisLinkId[4] = {-1,-1,-1,-1};
	  for(int kk=0; kk!=nModsInFile; kk++){
	    int modcount = 0;
	    if(only_path==I_name[kk]){
	      thisFedId[modcount] = I_fedId[kk];
	      thisLinkId[modcount] = I_linkId[kk];
	      modcount++;
	    }
	  }
	  for(int kk=0; kk!=4; kk++) allmodsMap->Fill(thisFedId[kk],thisLinkId[kk]);
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
	    //errmodsMap->Fill(structId,subStructId);
	    for(int kk=0; kk!=4; kk++) errmodsMap->Fill(thisFedId[kk],thisLinkId[kk]);
	  }
	}
      }
      int nbin = 0;
      if(full_path.find("Layer_1")!=string::npos) nbin = 1;
      else if(full_path.find("Layer_2")!=string::npos) nbin = 2;
      else if(full_path.find("Layer_3")!=string::npos) nbin = 3;
      else if(full_path.find("Disk_1")!=string::npos && full_path.find("HalfCylinder_m")!=string::npos) nbin = 4;
      else if(full_path.find("Disk_2")!=string::npos && full_path.find("HalfCylinder_m")!=string::npos) nbin = 5;
      else if(full_path.find("Disk_1")!=string::npos && full_path.find("HalfCylinder_p")!=string::npos) nbin = 6;
      else if(full_path.find("Disk_2")!=string::npos && full_path.find("HalfCylinder_p")!=string::npos) nbin = 7;
/*
//+++++++++++++ Fill NDigisMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("ndigis_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/NDigisCut");
        NDigisMap = bei->get("Pixel/EventInfo/reportSummaryContents/NDigisCut/NDigisMap");
        if(!NDigisMap) continue; 
	if(me->hasError() && me->getEntries()>50) NDigisMap->setBinContent(nbin,1,0.);
	else  NDigisMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill DigiChargeMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("adc_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/DigiChargeCut");
        DigiChargeMap = bei->get("Pixel/EventInfo/reportSummaryContents/DigiChargeCut/DigiChargeMap");
        if(!DigiChargeMap) continue; 
	if(me->hasError() && me->getEntries()>50) DigiChargeMap->setBinContent(nbin,1,0.);
	else  DigiChargeMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill OnTrackClusterSizeMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("size_OnTrack_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/OnTrackClusterSizeCut");
        OnTrackClusterSizeMap = bei->get("Pixel/EventInfo/reportSummaryContents/OnTrackClusterSizeCut/OnTrackClusterSizeMap");
        if(!OnTrackClusterSizeMap) continue; 
	if(me->hasError() && me->getEntries()>50) OnTrackClusterSizeMap->setBinContent(nbin,1,0.);
	else  OnTrackClusterSizeMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill OnTrackNClustersMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("nclusters_OnTrack_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/OnTrackNClustersCut");
        OnTrackNClustersMap = bei->get("Pixel/EventInfo/reportSummaryContents/OnTrackNClustersCut/OnTrackNClustersMap");
        if(!OnTrackNClustersMap) continue; 
	if(me->hasError() && me->getEntries()>50) OnTrackNClustersMap->setBinContent(nbin,1,0.);
	else  OnTrackNClustersMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill OnTrackClusterChargeMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("charge_OnTrack_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/OnTrackClusterChargeCut");
        OnTrackClusterChargeMap = bei->get("Pixel/EventInfo/reportSummaryContents/OnTrackClusterChargeCut/OnTrackClusterChargeMap");
        if(!OnTrackClusterChargeMap) continue; 
	if(me->hasError() && me->getEntries()>50) OnTrackClusterChargeMap->setBinContent(nbin,1,0.);
	else  OnTrackClusterChargeMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill OffTrackClusterSizeMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("size_OffTrack_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/OffTrackClusterSizeCut");
        OffTrackClusterSizeMap = bei->get("Pixel/EventInfo/reportSummaryContents/OffTrackClusterSizeCut/OffTrackClusterSizeMap");
        if(!OffTrackClusterSizeMap) continue; 
	if(me->hasError() && me->getEntries()>50) OffTrackClusterSizeMap->setBinContent(nbin,1,0.);
	else  OffTrackClusterSizeMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill OffTrackNClustersMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("nclusters_OffTrack_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/OffTrackNClustersCut");
        OffTrackNClustersMap = bei->get("Pixel/EventInfo/reportSummaryContents/OffTrackNClustersCut/OffTrackNClustersMap");
        if(!OffTrackNClustersMap) continue; 
	if(me->hasError() && me->getEntries()>50) OffTrackNClustersMap->setBinContent(nbin,1,0.);
	else  OffTrackNClustersMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill OffTrackClusterChargeMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("charge_OffTrack_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/OffTrackClusterChargeCut");
        OffTrackClusterChargeMap = bei->get("Pixel/EventInfo/reportSummaryContents/OffTrackClusterChargeCut/OffTrackClusterChargeMap");
        if(!OffTrackClusterChargeMap) continue; 
	if(me->hasError() && me->getEntries()>50) OffTrackClusterChargeMap->setBinContent(nbin,1,0.);
	else  OffTrackClusterChargeMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill ResidualXMeanMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("residualX_mean_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/ResidualXMeanCut");
        ResidualXMeanMap = bei->get("Pixel/EventInfo/reportSummaryContents/ResidualXMeanCut/ResidualXMeanMap");
        if(!ResidualXMeanMap) continue; 
	if(me->hasError() && me->getEntries()>50) ResidualXMeanMap->setBinContent(nbin,1,0.);
	else  ResidualXMeanMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill ResidualXRMSMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("residualX_RMS_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/ResidualXRMSCut");
        ResidualXRMSMap = bei->get("Pixel/EventInfo/reportSummaryContents/ResidualXRMSCut/ResidualXRMSMap");
        if(!ResidualXRMSMap) continue; 
	if(me->hasError() && me->getEntries()>50) ResidualXRMSMap->setBinContent(nbin,1,0.);
	else  ResidualXRMSMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill ResidualYMeanMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("residualY_mean_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/ResidualYMeanCut");
        ResidualYMeanMap = bei->get("Pixel/EventInfo/reportSummaryContents/ResidualYMeanCut/ResidualYMeanMap");
        if(!ResidualYMeanMap) continue; 
	if(me->hasError() && me->getEntries()>50) ResidualYMeanMap->setBinContent(nbin,1,0.);
	else  ResidualYMeanMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill ResidualYRMSMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("residualY_RMS_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/ResidualYRMSCut");
        ResidualYRMSMap = bei->get("Pixel/EventInfo/reportSummaryContents/ResidualYRMSCut/ResidualYRMSMap");
        if(!ResidualYRMSMap) continue; 
	if(me->hasError() && me->getEntries()>50) ResidualYRMSMap->setBinContent(nbin,1,0.);
	else  ResidualYRMSMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill RecHitErrorXMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("ErrorX_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/RecHitErrorXCut");
        RecHitErrorXMap = bei->get("Pixel/EventInfo/reportSummaryContents/RecHitErrorXCut/RecHitErrorXMap");
        if(!RecHitErrorXMap) continue; 
	if(me->hasError() && me->getEntries()>50) RecHitErrorXMap->setBinContent(nbin,1,0.);
	else  RecHitErrorXMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++ Fill RecHitErrorYMap:     
      if(full_path.find("SUM")==string::npos && full_path.find("ErrorY_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
        bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/RecHitErrorYCut");
        RecHitErrorYMap = bei->get("Pixel/EventInfo/reportSummaryContents/RecHitErrorYCut/RecHitErrorYMap");
        if(!RecHitErrorYMap) continue; 
	if(me->hasError() && me->getEntries()>50) RecHitErrorYMap->setBinContent(nbin,1,0.);
	else  RecHitErrorYMap->setBinContent(nbin,1,1.);
      }
//+++++++++++++++++++++===================      
*/    }
  }
  vector<string> subDirVec = bei->getSubdirs();  
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    bei->cd(*ic);
    init=false;
    fillGlobalQualityPlot(bei,init,eSetup,nFEDs,Tier0Flag);
    bei->goUp();
  }
  bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents/FEDErrorsCut");
  NErrorsFEDMap = bei->get("Pixel/EventInfo/reportSummaryContents/FEDErrorsCut/NErrorsFEDMap");
  if(NErrorsFEDMap){ 
    float contents=0.;
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
      NErrorsFEDMap->setBinContent(i,j,contents);
    }
  }
  if(allmodsMap) allmodsMap->Clear();
  if(goodmodsMap) goodmodsMap->Clear();
  if(errmodsMap) errmodsMap->Clear();

//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

/*  if(bei->pwd() == "Pixel/EventInfo/reportSummaryContents"){
    SummaryReportMap = bei->get("Pixel/EventInfo/reportSummaryMap");
    if(SummaryReportMap){ 
      float contents=0.;
      if(!Tier0Flag){
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
      }else{
        for(int i=1; i!=8; i++)for(int j=1; j!=23; j++){
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
      }
    }
  }*/
  

  //cout<<"counters: "<<count<<" , "<<errcount<<endl;
}

//**********************************************************************************************

