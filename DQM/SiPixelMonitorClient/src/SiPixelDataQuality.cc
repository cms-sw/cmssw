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
  if(allmodsMap) delete allmodsMap;
  if(errmodsMap) delete errmodsMap;
  if(goodmodsMap) delete goodmodsMap;
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

void SiPixelDataQuality::bookGlobalQualityFlag(DQMStore * bei, bool Tier0Flag, int nFEDs) {
//std::cout<<"BOOK GLOBAL QUALITY FLAG MEs!"<<std::endl;
  bei->cd();
  
  bei->setCurrentFolder("Pixel/EventInfo");
  if(!Tier0Flag){
    SummaryReportMap = bei->book2D("reportSummaryMap","Pixel Summary Map",40,0.,40.,36,1.,37.);
    SummaryReportMap->setAxisTitle("Pixel FED #",1);
    SummaryReportMap->setAxisTitle("Pixel FED Channel #",2);
    allmodsMap = new TH2F("allmodsMap","allmodsMap",40,0.,40.,36,1.,37.);
    errmodsMap = new TH2F("errmodsMap","errmodsMap",40,0.,40.,36,1.,37.);
    goodmodsMap = new TH2F("goodmodsMap","goodmodsMap",40,0.,40.,36,1.,37.);
  }else{
    SummaryReportMap = bei->book2D("reportSummaryMap","Pixel Summary Map",2,0.,2.,7,0.,7.);
    SummaryReportMap->setBinLabel(1,"Barrel",1);
    SummaryReportMap->setBinLabel(2,"Endcaps",1);
    SummaryReportMap->setBinLabel(1,"No errors",2);
    SummaryReportMap->setBinLabel(2,"Pass ndigis cut",2);
    SummaryReportMap->setBinLabel(3,"Pass digi charge cut",2);
    SummaryReportMap->setBinLabel(4,"Pass OnTrack cluster size cut",2);
    SummaryReportMap->setBinLabel(5,"Pass OnTrack nclusters cut",2);
    SummaryReportMap->setBinLabel(6,"Pass OnTrack cluster charge",2);
    SummaryReportMap->setBinLabel(7,"Pass hiteff cut",2);
    allmodsMap = new TH2F("allmodsMap","allmodsMap",2,0.,2.,7,0.,7.);
    errmodsMap = new TH2F("errmodsMap","errmodsMap",2,0.,2.,7,0.,7.);
    goodmodsMap = new TH2F("goodmodsMap","goodmodsMap",2,0.,2.,7,0.,7.);
  }  
    SummaryPixel = bei->bookFloat("reportSummary");
  bei->setCurrentFolder("Pixel/EventInfo/reportSummaryContents");
    SummaryBarrel = bei->bookFloat("PixelBarrelFraction");
    SummaryEndcap = bei->bookFloat("PixelEndcapFraction");
  // book the data certification cuts:
  bei->setCurrentFolder("Pixel/AdditionalPixelErrors");
    NErrorsFEDs = bei->bookFloat("FEDsNErrorsCut");
  bei->setCurrentFolder("Pixel/Barrel");
    NErrorsBarrel = bei->bookFloat("BarrelNErrorsCut");
    NDigisBarrel = bei->bookInt("BarrelNDigisCut");
    DigiChargeBarrel = bei->bookInt("BarrelDigiChargeCut");
    ClusterSizeBarrel = bei->bookInt("BarrelClusterSizeCut");
    NClustersBarrel = bei->bookInt("BarrelNClustersCut");
    ClusterChargeBarrel = bei->bookInt("BarrelClusterChargeCut");
  bei->setCurrentFolder("Pixel/Endcap");
    NErrorsEndcap = bei->bookFloat("EndcapNErrorsCut");
    NDigisEndcap = bei->bookInt("EndcapNDigisCut");
    DigiChargeEndcap = bei->bookInt("EndcapDigiChargeCut");
    ClusterSizeEndcap = bei->bookInt("EndcapClusterSizeCut");
    NClustersEndcap = bei->bookInt("EndcapNClustersCut");
    ClusterChargeEndcap = bei->bookInt("EndcapClusterChargeCut");
  if(Tier0Flag){
    bei->setCurrentFolder("Pixel/Tracks");
      NPixelTracks = bei->bookInt("PixelTracksCut");
  }
    
    // Init MonitoringElements:
    if(nFEDs>0){
      SummaryPixel = bei->get("Pixel/EventInfo/reportSummary");
      if(SummaryPixel) SummaryPixel->Fill(1.);
      SummaryBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/PixelBarrelFraction");
      if(SummaryBarrel) SummaryBarrel->Fill(1.);
      SummaryEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/PixelEndcapFraction");
      if(SummaryEndcap)	SummaryEndcap->Fill(1.);
    }else{
      SummaryPixel = bei->get("Pixel/EventInfo/reportSummary");
      if(SummaryPixel) SummaryPixel->Fill(-1.);
      SummaryBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/PixelBarrelFraction");
      if(SummaryBarrel) SummaryBarrel->Fill(-1.);
      SummaryEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/PixelEndcapFraction");
      if(SummaryEndcap)	SummaryEndcap->Fill(-1.);
    }
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
    ClusterSizeBarrel = bei->get("Pixel/Barrel/BarrelClusterSizeCut");
    if(ClusterSizeBarrel) ClusterSizeBarrel->Fill(1);
    ClusterSizeEndcap = bei->get("Pixel/Endcap/EndcapClusterSizeCut");
    if(ClusterSizeEndcap) ClusterSizeEndcap->Fill(1);
    ClusterChargeBarrel = bei->get("Pixel/Barrel/BarrelClusterChargeCut");
    if(ClusterChargeBarrel) ClusterChargeBarrel->Fill(1);
    ClusterChargeEndcap = bei->get("Pixel/Endcap/EndcapClusterChargeCut");
    if(ClusterChargeEndcap) ClusterChargeEndcap->Fill(1);
    NClustersBarrel = bei->get("Pixel/Barrel/BarrelNClustersCut");
    if(NClustersBarrel) NClustersBarrel->Fill(1);
    NClustersEndcap = bei->get("Pixel/Endcap/EndcapNClustersCut");
    if(NClustersEndcap) NClustersEndcap->Fill(1);
    if(Tier0Flag){
      NPixelTracks = bei->get("Pixel/Tracks/PixelTracksCut");
      if(NPixelTracks) NPixelTracks->Fill(1);
    }
   
    SummaryReportMap = bei->get("Pixel/EventInfo/reportSummaryMap");
    if(SummaryReportMap){
      if(!Tier0Flag) for(int i=1; i!=41; i++) for(int j=1; j!=37; j++) SummaryReportMap->setBinContent(i,j,-1.);
      if(Tier0Flag) for(int i=1; i!=3; i++) for(int j=1; j!=8; j++) SummaryReportMap->setBinContent(i,j,-1.);
    }
    if(!Tier0Flag){
      for(int i=1; i!=41; i++) for(int j=1; j!=37; j++){
        if(allmodsMap) allmodsMap->SetBinContent(i,j,0.);
        if(errmodsMap) errmodsMap->SetBinContent(i,j,0.);
        if(goodmodsMap) goodmodsMap->SetBinContent(i,j,0.);
      }
    }
    if(Tier0Flag){
      for(int i=1; i!=3; i++) for(int j=1; j!=8; j++){
        if(allmodsMap) allmodsMap->SetBinContent(i,j,0.);
        if(errmodsMap) errmodsMap->SetBinContent(i,j,0.);
        if(goodmodsMap) goodmodsMap->SetBinContent(i,j,0.);
      }
    }
    
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
//cout<<"Entering SiPixelDataQuality::computeGlobalQualityFlag for the first time"<<endl;
    allMods_=0; errorMods_=0; qflag_=0.; 
    barrelMods_=0; endcapMods_=0;
    objectCount_=0;
    DONE_ = false;
    
    //Error counters and flags:
    n_errors_barrel_=0; barrel_error_flag_=0.;
    n_errors_endcap_=0; endcap_error_flag_=0.;
    n_errors_feds_=0; feds_error_flag_=0.;
    digiStatsBarrel = false, clusterStatsBarrel = false, trackStatsBarrel = false;
    digiCounterBarrel = 0, clusterCounterBarrel = 0, trackCounterBarrel = 0;
    digiStatsEndcap = false, clusterStatsEndcap = false, trackStatsEndcap = false;
    digiCounterEndcap = 0, clusterCounterEndcap = 0, trackCounterEndcap = 0;
    init=false;
  }
  if(nFEDs==0) return;  
  
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
//cout<<"currDir="<<currDir<<endl;

  if((!Tier0Flag && dname.find("Module_")!=string::npos) || 
     (Tier0Flag && (dname.find("Ladder_")!=string::npos || dname.find("Blade_")!=string::npos))){

    objectCount_++;

    if(currDir.find("Pixel")!=string::npos) allMods_++;
    if(currDir.find("Barrel")!=string::npos) barrelMods_++;
    if(currDir.find("Endcap")!=string::npos) endcapMods_++;
    vector<string> meVec = bei->getMEs();
    for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
      string full_path = currDir + "/" + (*it);
      //cout<<"full_path:"<<full_path<<endl;
      if(full_path.find("NErrors_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getMean()>0){
	//cout<<"found a module with errors: "<<full_path<<endl;
	  full_path = full_path.replace(full_path.find("NErrors"),7,"errorType");
	  //cout<<"changing histo name to: "<<full_path<<endl;
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
	    full_path = full_path.replace(full_path.find("errorType"),9,"TBMMessage");
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
	  //cout<<"errmod counters: "<<errorMods_<<","<<n_errors_barrel_<<","<<n_errors_endcap_<<","<<n_errors_barrelL1_<<","<<n_errors_barrelL2_<<","<<n_errors_barrelL3_<<endl;
        }	
      }else if(full_path.find("ndigis_")!=string::npos){
      //cout<<"found an ndigi histo now"<<endl;
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
	//cout<<"got the histo now"<<endl;
        if(me->getEntries()>25){
	//cout<<"histo has more than 50 entries"<<endl;
	  if(full_path.find("Barrel")!=string::npos) digiCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) digiCounterEndcap++;
	  //cout<<"counter are now: "<<digiCounterBarrel<<","<<digiCounterBarrelL1<<endl;
        }
      }else if(Tier0Flag && full_path.find("nclusters_OnTrack_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>25){
	  if(full_path.find("Barrel")!=string::npos) clusterCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) clusterCounterEndcap++;
        }
      }else if(!Tier0Flag && full_path.find("nclusters_")!=string::npos){
        MonitorElement * me = bei->get(full_path);
        if(!me) continue;
        if(me->getEntries()>25){
	  if(full_path.find("Barrel")!=string::npos) clusterCounterBarrel++;
	  if(full_path.find("Endcap")!=string::npos) clusterCounterEndcap++;
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
  NErrorsBarrel = bei->get("Pixel/Barrel/BarrelNErrorsCut");
  if(NErrorsBarrel) NErrorsBarrel->Fill(barrel_error_flag_);
  NErrorsEndcap = bei->get("Pixel/Endcap/EndcapNErrorsCut");
  if(NErrorsEndcap)   NErrorsEndcap->Fill(endcap_error_flag_);
  NErrorsFEDs = bei->get("Pixel/AdditionalPixelErrors/FEDsNErrorsCut");
  if(NErrorsFEDs) NErrorsFEDs->Fill(1.); // hardwired for the moment, need to fix!

  
  string meName0;
  MonitorElement * me;
  
  // Fill the Digi flags:
  if(!Tier0Flag){
    meName0 = "Pixel/Barrel/SUMDIG_ndigis_Barrel";
    if(digiCounterBarrel/768 > 0.5) digiStatsBarrel = true;
    if(digiCounterEndcap/672 > 0.5) digiStatsEndcap = true;
    //cout<<"digiStatsBarrel="<<digiStatsBarrel<<" , digiStatsEndcap="<<digiStatsEndcap<<endl;
  }else{
    meName0 = "Pixel/Barrel/SUMOFF_ndigis_Barrel"; 
    if(digiCounterBarrel/192 > 0.5) digiStatsBarrel = true;
    if(digiCounterEndcap/96 > 0.5) digiStatsEndcap = true;
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
     
     
    // Fill the Cluster flags:
  if(!Tier0Flag){
    meName0 = "Pixel/Barrel/SUMCLU_size_Barrel";
    if(clusterCounterBarrel/768 > 0.5) clusterStatsBarrel = true;
    if(clusterCounterEndcap/672 > 0.5) clusterStatsEndcap = true;
  }else{
    meName0 = "Pixel/Barrel/SUMOFF_size_OnTrack_Barrel"; 
    if(clusterCounterBarrel/192 > 0.5) clusterStatsBarrel = true;
    if(clusterCounterEndcap/96 > 0.5) clusterStatsEndcap = true;
  }
  me = bei->get(meName0);
  if(me){
    ClusterSizeBarrel = bei->get("Pixel/Barrel/BarrelClusterSizeCut");
    if(ClusterSizeBarrel && clusterStatsBarrel){
      if(me->hasError()) ClusterSizeBarrel->Fill(0);
      else ClusterSizeBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMCLU_size_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_size_OnTrack_Endcap"; 
  me = bei->get(meName0);
  if(me){
    ClusterSizeEndcap = bei->get("Pixel/Endcap/EndcapClusterSizeCut");
    if(ClusterSizeEndcap && clusterStatsEndcap){
      if(me->hasError()) ClusterSizeEndcap->Fill(0);
      else ClusterSizeEndcap->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Barrel/SUMCLU_charge_Barrel";
  else meName0 = "Pixel/Barrel/SUMOFF_charge_OnTrack_Barrel"; 
  me = bei->get(meName0);
  if(me){
    ClusterChargeBarrel = bei->get("Pixel/Barrel/BarrelClusterChargeCut");
    if(ClusterChargeBarrel && clusterStatsBarrel){
      if(me->hasError()) ClusterChargeBarrel->Fill(0);
      else ClusterChargeBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMCLU_charge_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_charge_OnTrack_Endcap"; 
  me = bei->get(meName0);
  if(me){
    ClusterChargeEndcap = bei->get("Pixel/Endcap/EndcapClusterChargeCut");
    if(ClusterChargeEndcap && clusterStatsEndcap){
      if(me->hasError()) ClusterChargeEndcap->Fill(0);
      else ClusterChargeEndcap->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Barrel/SUMCLU_nclusters_Barrel";
  else meName0 = "Pixel/Barrel/SUMOFF_nclusters_OnTrack_Barrel"; 
  me = bei->get(meName0);
  if(me){
    NClustersBarrel = bei->get("Pixel/Barrel/BarrelNClustersCut");
    if(NClustersBarrel && clusterStatsBarrel){
      if(me->hasError()) NClustersBarrel->Fill(0);
      else NClustersBarrel->Fill(1);
    }
  }
  if(!Tier0Flag) meName0 = "Pixel/Endcap/SUMCLU_nclusters_Endcap";
  else meName0 = "Pixel/Endcap/SUMOFF_nclusters_OnTrack_Endcap"; 
  me = bei->get(meName0);
  if(me){
    NClustersEndcap = bei->get("Pixel/Endcap/EndcapNClustersCut");
    if(NClustersEndcap && clusterStatsEndcap){
      if(me->hasError()) NClustersEndcap->Fill(0);
      else NClustersEndcap->Fill(1);
    }
  }
  // Pixel Track multiplicity / Pixel hit efficiency
  meName0 = "Pixel/Tracks/ntracks_generalTracks";
  me = bei->get(meName0);
  if(me){
    NPixelTracks = bei->get("Pixel/Tracks/PixelTracksCut");
    if(NPixelTracks && me->getBinContent(1)>1000){
      if((float)me->getBinContent(2)/(float)me->getBinContent(1)<0.01){
        NPixelTracks->Fill(0);
      }else{ 
        NPixelTracks->Fill(1);
      }
    }
  }
  
  
//********************************************************************************************************  
  
  // Final combination of all Data Quality results:
  float pixelFlag = -1., barrelFlag = -1., endcapFlag = -1.;
  float barrel_errors_temp[1]={-1.}; int barrel_cuts_temp[5]={5*-1}; 
  float endcap_errors_temp[1]={-1.}; int endcap_cuts_temp[5]={5*-1}; 
  int pixel_cuts_temp[1]={-1};
  float combinedCuts = 1.; int numerator = 0, denominator = 0;

  // Barrel results:
  me = bei->get("Pixel/Barrel/BarrelNErrorsCut");
  if(me) barrel_errors_temp[0] = me->getFloatValue();
  me = bei->get("Pixel/Barrel/BarrelNDigisCut");
  if(me) barrel_cuts_temp[0] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelDigiChargeCut");
  if(me) barrel_cuts_temp[1] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelClusterSizeCut");
  if(me) barrel_cuts_temp[2] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelNClustersCut");
  if(me) barrel_cuts_temp[3] = me->getIntValue();
  me = bei->get("Pixel/Barrel/BarrelClusterChargeCut");
  if(me) barrel_cuts_temp[4] = me->getIntValue();
  for(int k=0; k!=5; k++){
    if(barrel_cuts_temp[k]>=0){
      numerator = numerator + barrel_cuts_temp[k];
      denominator++;
      //cout<<"cut flag, Barrel: "<<k<<","<<barrel_cuts_temp[k]<<","<<numerator<<","<<denominator<<endl;
    }
  } 
  if(denominator!=0) combinedCuts = float(numerator)/float(denominator);  
  barrelFlag = barrel_errors_temp[0] * combinedCuts;
  
  
  //cout<<" the resulting barrel flag is: "<<barrel_errors_temp[0]<<"*"<<combinedCuts<<"="<<barrelFlag<<endl;
  
  // Endcap results:
  combinedCuts = 1.; numerator = 0; denominator = 0;
  me = bei->get("Pixel/Endcap/EndcapNErrorsCut");
  if(me) endcap_errors_temp[0] = me->getFloatValue();
  me = bei->get("Pixel/Endcap/EndcapNDigisCut");
  if(me) endcap_cuts_temp[0] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapDigiChargeCut");
  if(me) endcap_cuts_temp[1] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapClusterSizeCut");
  if(me) endcap_cuts_temp[2] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapNClustersCut");
  if(me) endcap_cuts_temp[3] = me->getIntValue();
  me = bei->get("Pixel/Endcap/EndcapClusterChargeCut");
  if(me) endcap_cuts_temp[4] = me->getIntValue();
  for(int k=0; k!=5; k++){
    if(endcap_cuts_temp[k]>=0){
      numerator = numerator + endcap_cuts_temp[k];
      denominator++;
      //cout<<"cut flag, Endcap: "<<k<<","<<endcap_cuts_temp[k]<<","<<numerator<<","<<denominator<<endl;
    }
  } 
  if(denominator!=0) combinedCuts = float(numerator)/float(denominator);  
  endcapFlag = endcap_errors_temp[0] * combinedCuts;
  //cout<<" the resulting endcap flag is: "<<endcap_errors_temp[0]<<"*"<<combinedCuts<<"="<<endcapFlag<<endl;
  
  // Track results:
  combinedCuts = 1.; numerator = 0; denominator = 0;
  me = bei->get("Pixel/Tracks/PixelTracksCut");
  if(me) pixel_cuts_temp[0] = me->getIntValue();

  //Combination of all:
  combinedCuts = 1.; numerator = 0; denominator = 0;
  for(int k=0; k!=5; k++){
    if(barrel_cuts_temp[k]>=0){
      numerator = numerator + barrel_cuts_temp[k];
      denominator++;
    }
    //cout<<"after barrel: num="<<numerator<<" , den="<<denominator<<endl;
    if(endcap_cuts_temp[k]>=0){
      numerator = numerator + endcap_cuts_temp[k];
      denominator++;
    }
    if(k<1 && pixel_cuts_temp[k]>=0){
      numerator = numerator + pixel_cuts_temp[k];
      denominator++;
    }
    //cout<<"after both: num="<<numerator<<" , den="<<denominator<<endl;
  } 
  if(denominator!=0) combinedCuts = float(numerator)/float(denominator); 
  pixelFlag = float(barrelMods_-n_errors_barrel_+endcapMods_-n_errors_endcap_)/float(barrelMods_+endcapMods_) * float(combinedCuts);
  
  
  //cout<<"barrel, endcap, pixel flags: "<<barrelFlag<<","<<endcapFlag<<","<<pixelFlag<<endl;
  SummaryPixel = bei->get("Pixel/EventInfo/reportSummary");
  if(SummaryPixel) SummaryPixel->Fill(pixelFlag);
  SummaryBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/PixelBarrelFraction");
  if(SummaryBarrel) SummaryBarrel->Fill(barrelFlag);
  SummaryEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/PixelEndcapFraction");
  if(SummaryEndcap)   SummaryEndcap->Fill(endcapFlag);
  }
}

//**********************************************************************************************

void SiPixelDataQuality::fillGlobalQualityPlot(DQMStore * bei, bool init, edm::EventSetup const& eSetup, int nFEDs, bool Tier0Flag)
{
  //calculate eta and phi of the modules and fill a 2D plot:
  if(init){
    count=0; errcount=0;
    init=false;
  }
  
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
// Fill Maps:
  // Online:    
  if(nFEDs==0) return;
  eSetup.get<SiPixelFedCablingMapRcd>().get(theCablingMap);
  string currDir = bei->pwd();
  if(currDir.find("Reference")!=string::npos) return;
  //cout<<"currDir="<<currDir<<endl;
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  // find a detId for Blades and Ladders (first of the contained Modules!):
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
  if(!Tier0Flag){
  if(dname.find("Module_")!=string::npos && currDir.find("Reference")==string::npos){
    vector<string> meVec = bei->getMEs();
    int detId=-1; int fedId=-1; int linkId=-1;
    for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
      //checking for any digis or FED errors to decide if this module is in DAQ:  
      string full_path = currDir + "/" + (*it);
      //cout<<"path: "<<full_path<<" , detId= "<<detId<<endl;
      if(detId==-1 && full_path.find("SUMOFF")==string::npos &&
         ((full_path.find("ndigis")!=string::npos && full_path.find("SUMDIG")==string::npos) || 
	  (full_path.find("NErrors")!=string::npos && full_path.find("SUMRAW")==string::npos)) && 
	  (getDetId(bei->get(full_path)) > 100)){
	//cout<<"Got into the first ndigis or NErrors histogram!"<<endl;
        MonitorElement * me = bei->get(full_path);
        if (!me) continue;
	if((full_path.find("ndigis")!=string::npos && me->getMean()>0.) ||
	   (full_path.find("NErrors")!=string::npos && me->getMean()>0.)){ 
	  //cout<<"found a module with digis or errors: "<<full_path<<endl;
          detId = getDetId(me);
	  //cout<<"got a module with digis or errors and the detid is: "<<detId<<endl;
          for(int fedid=0; fedid!=40; ++fedid){
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
	  //cout<<"it has this FED ID and channel ID: "<<fedId<<" , "<<linkId<<endl;
	  allmodsMap->Fill(fedId,linkId);
	  //cout<<"this is a module that has digis and/or errors: "<<detId<<","<<fedId<<","<<linkId<<endl;
	  //use presence of any FED error as error flag (except for TBM or ROC resets):
          bool anyerr=false; bool type30=false; bool othererr=false;
          if(full_path.find("ndigis")!=string::npos) full_path = full_path.replace(full_path.find("ndigis"),6,"NErrors");
	  me = bei->get(full_path);
	  if(me && me->getMean()>0.) anyerr=true;
          //cout<<"here is an error: "<<detId<<","<<me->getMean()<<endl;
	  if(full_path.find("NErrors")!=string::npos) full_path = full_path.replace(full_path.find("NErrors"),7,"errorType");
	  me = bei->get(full_path);
	  if(me){
	    for(int jj=1; jj<16; jj++){
	    //cout<<"looping over errorType: "<<jj<<" , "<<me->getBinContent(jj)<<endl;
	      if(me->getBinContent(jj)>0.){
	        if(jj!=6) othererr=true;
		else type30=true;
	      }
	    }
	    if(type30){
	      full_path = full_path.replace(full_path.find("errorType"),9,"TBMMessage");
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
  }// end ONLINE
  
  //cout<<"Loop over all modules is done, now I am in    "<<bei->pwd()<<"     and currDir is    "<<currDir<<endl;
  bei->cd("Pixel/EventInfo/reportSummaryContents");
  //cout<<"Loop over all modules is done, now I am in    "<<bei->pwd()<<"     and currDir is    "<<currDir<<endl;
  if(bei->pwd()=="Pixel/EventInfo/reportSummaryContents"){
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
        float barrel_errors_temp[1]={-1.}; int barrel_cuts_temp[6]={6*-1}; 
        float endcap_errors_temp[1]={-1.}; int endcap_cuts_temp[6]={6*-1}; 
        int pixel_cuts_temp[1]={-1};
        // Barrel results:
        MonitorElement * me;
	me = bei->get("Pixel/Barrel/BarrelNErrorsCut");
        if(me) barrel_errors_temp[0] = me->getFloatValue();
	me = bei->get("Pixel/Endcap/EndcapNErrorsCut");
        if(me) endcap_errors_temp[0] = me->getFloatValue();
        SummaryReportMap->setBinContent(1,1,barrel_errors_temp[0]);
        SummaryReportMap->setBinContent(2,1,endcap_errors_temp[0]);
        me = bei->get("Pixel/Barrel/BarrelNDigisCut");
        if(me) barrel_cuts_temp[0] = me->getIntValue();
        me = bei->get("Pixel/Barrel/BarrelDigiChargeCut");
        if(me) barrel_cuts_temp[1] = me->getIntValue();
        me = bei->get("Pixel/Barrel/BarrelClusterSizeCut");
        if(me) barrel_cuts_temp[2] = me->getIntValue();
        me = bei->get("Pixel/Barrel/BarrelNClustersCut");
        if(me) barrel_cuts_temp[3] = me->getIntValue();
        me = bei->get("Pixel/Barrel/BarrelClusterChargeCut");
        if(me) barrel_cuts_temp[4] = me->getIntValue();  
        me = bei->get("Pixel/Endcap/EndcapNDigisCut");
        if(me) endcap_cuts_temp[0] = me->getIntValue();
        me = bei->get("Pixel/Endcap/EndcapDigiChargeCut");
        if(me) endcap_cuts_temp[1] = me->getIntValue();
        me = bei->get("Pixel/Endcap/EndcapClusterSizeCut");
        if(me) endcap_cuts_temp[2] = me->getIntValue();
        me = bei->get("Pixel/Endcap/EndcapNClustersCut");
        if(me) endcap_cuts_temp[3] = me->getIntValue();
        me = bei->get("Pixel/Endcap/EndcapClusterChargeCut");
        if(me) endcap_cuts_temp[4] = me->getIntValue();  
        for(int j=2; j!=7; j++){
          SummaryReportMap->setBinContent(1,j,barrel_cuts_temp[j-2]);
          SummaryReportMap->setBinContent(2,j,endcap_cuts_temp[j-2]);
	  //cout<<"error cut values: "<<j<<" , "<<barrel_cuts_temp[j-2]<<" , "<<endcap_cuts_temp[j-2]<<endl;
        }
        me = bei->get("Pixel/Tracks/PixelTracksCut");
        if(me) pixel_cuts_temp[0] = me->getIntValue();  
        SummaryReportMap->setBinContent(1,7,pixel_cuts_temp[0]);
        SummaryReportMap->setBinContent(2,7,pixel_cuts_temp[0]);
      }
    }
    if(allmodsMap) allmodsMap->Clear();
    if(goodmodsMap) goodmodsMap->Clear();
    if(errmodsMap) errmodsMap->Clear();
  }

  //cout<<"counters: "<<count<<" , "<<errcount<<endl;
}

//**********************************************************************************************

