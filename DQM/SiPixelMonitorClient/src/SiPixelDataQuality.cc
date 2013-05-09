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

  allmodsMap=0;
  errmodsMap=0;
  goodmodsMap=0;
  allmodsVec=0;
  errmodsVec=0;
  goodmodsVec=0;
  for (int i = 0; i < 40; ++i)
    {lastallmods_[i] = 0; lasterrmods_[i] = 0;}
  timeoutCounter_=0;
  lastLS_=-1;
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
  if(allmodsVec) delete allmodsVec;
  if(errmodsVec) delete errmodsVec;
  if(goodmodsVec) delete goodmodsVec;
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

  bei->setCurrentFolder("Pixel/Barrel");
  if(!Tier0Flag){
    ClusterModAll = bei->book1D("NClustertoChargeRatio_AllMod","Cluster Noise All Modules", 768, 0., 768.);
    ClusterMod1   = bei->book1D("NClustertoChargeRatio_NormMod1",  "Normalized N_{Clusters} to Charge Ratio per Module1", 192, 0., 192.);
    ClusterMod2   = bei->book1D("NClustertoChargeRatio_NormMod2",  "Normalized N_{Clusters} to Charge Ratio per Module2", 192, 0., 192.);
    ClusterMod3   = bei->book1D("NClustertoChargeRatio_NormMod3",  "Normalized N_{Clusters} to Charge Ratio per Module3", 192, 0., 192.);
    ClusterMod4   = bei->book1D("NClustertoChargeRatio_NormMod4",  "Normalized N_{Clusters} to Charge Ratio per Module4", 192, 0., 192.);
  } 
  bei->setCurrentFolder("Pixel/EventInfo");
  if(!Tier0Flag){
    /*SummaryReportMap = bei->book2D("reportSummaryMap","Pixel Summary Map",40,0.,40.,36,1.,37.);
    SummaryReportMap->setAxisTitle("Pixel FED #",1);
    SummaryReportMap->setAxisTitle("Pixel FED Channel #",2);
    allmodsMap = new TH2F("allmodsMap","allmodsMap",40,0.,40.,36,1.,37.);
    errmodsMap = new TH2F("errmodsMap","errmodsMap",40,0.,40.,36,1.,37.);
    goodmodsMap = new TH2F("goodmodsMap","goodmodsMap",40,0.,40.,36,1.,37.);
    */
    SummaryReportMap = bei->book2D("reportSummaryMap","Pixel Summary Map",3000,0.,3000.,40,0.,40.);
    SummaryReportMap->setAxisTitle("Lumi Section",1);
    SummaryReportMap->setAxisTitle("Pixel FED #",2);
    allmodsVec = new TH1D("allmodsVec","allmodsVec",40,0.,40.);
    errmodsVec = new TH1D("errmodsVec","errmodsVec",40,0.,40.);
    goodmodsVec = new TH1D("goodmodsVec","goodmodsVec",40,0.,40.);
  }else{
    SummaryReportMap = bei->book2D("reportSummaryMap","Pixel Summary Map",2,0.,2.,7,0.,7.);
    SummaryReportMap->setBinLabel(1,"Barrel",1);
    SummaryReportMap->setBinLabel(2,"Endcaps",1);
    SummaryReportMap->setBinLabel(1,"Errors",2);
    SummaryReportMap->setBinLabel(2,"NDigis",2);
    SummaryReportMap->setBinLabel(3,"DigiCharge",2);
    SummaryReportMap->setBinLabel(4,"ClusterSize",2);
    SummaryReportMap->setBinLabel(5,"NClusters",2);
    SummaryReportMap->setBinLabel(6,"ClusterCharge",2);
    SummaryReportMap->setBinLabel(7,"HitEff",2);
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
      if(!Tier0Flag) for(int i=1; i!=3001; i++) for(int j=1; j!=41; j++) SummaryReportMap->setBinContent(i,j,-1.);
      if(Tier0Flag) for(int i=1; i!=3; i++) for(int j=1; j!=8; j++) SummaryReportMap->setBinContent(i,j,-1.);
    }
    if(!Tier0Flag){
      for(int j=1; j!=41; j++){
        if(allmodsVec) allmodsVec->SetBinContent(j,0.);
        if(errmodsVec) errmodsVec->SetBinContent(j,0.);
        if(goodmodsVec) goodmodsVec->SetBinContent(j,0.);
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
							   bool Tier0Flag){
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
    n_errors_pixel_=0; pixel_error_flag_=0.;
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
      if(full_path.find("ndigis_")!=string::npos){
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

  // Evaluate error flag now, only stored in AdditionalPixelErrors:
  MonitorElement * me_err = bei->get("Pixel/AdditionalPixelErrors/FedETypeNErrArray");
  MonitorElement * me_evt = bei->get("Pixel/EventInfo/processedEvents");
  if(me_err && me_evt){
    for(int i=1; i!=41; i++)for(int j=1; j!=22; j++)
      if(me_err->getBinContent(i,j)>0){
        n_errors_pixel_=n_errors_pixel_+int(me_err->getBinContent(i,j));
        if(i<33) n_errors_barrel_=n_errors_barrel_+int(me_err->getBinContent(i,j));
        if(i>32) n_errors_endcap_=n_errors_endcap_+int(me_err->getBinContent(i,j));
      }
    int NProcEvts = me_evt->getIntValue();
    if(NProcEvts>0){
      barrel_error_flag_ = (float(NProcEvts)-float(n_errors_barrel_))/float(NProcEvts);
      endcap_error_flag_ = (float(NProcEvts)-float(n_errors_endcap_))/float(NProcEvts);
      pixel_error_flag_ = (float(NProcEvts)-float(n_errors_barrel_)-float(n_errors_endcap_))/float(NProcEvts);
    }
  }
  NErrorsBarrel = bei->get("Pixel/Barrel/BarrelNErrorsCut");
  if(NErrorsBarrel) NErrorsBarrel->Fill(barrel_error_flag_);
  NErrorsEndcap = bei->get("Pixel/Endcap/EndcapNErrorsCut");
  if(NErrorsEndcap)   NErrorsEndcap->Fill(endcap_error_flag_);
  
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
  pixelFlag = float(pixel_error_flag_) * float(combinedCuts);
  
  
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

void SiPixelDataQuality::computeGlobalQualityFlagByLumi(DQMStore * bei, 
                                                           bool init,
							   int nFEDs,
							   bool Tier0Flag,
							   int nEvents_lastLS_,
							   int nErrorsBarrel_lastLS_,
							   int nErrorsEndcap_lastLS_){
//cout<<"entering SiPixelDataQuality::ComputeGlobalQualityFlagByLumi"<<endl;
//   cout << ACRed << ACBold
//        << "[SiPixelDataQuality::ComputeGlobalQualityFlag]"
//        << ACPlain
//        << " Enter" 
//        << endl ;

  if(nFEDs==0) return;  
  
  // evaluate fatal FED errors for data quality:
  float BarrelRate_LS = 1.;
  float EndcapRate_LS = 1.;
  float PixelRate_LS = 1.;
  MonitorElement * me = bei->get("Pixel/AdditionalPixelErrors/byLumiErrors");
  if(me){
    //cout<<"NENTRIES: "<<me->getEntries()<<" "<<nEvents_lastLS_<<" "<<nErrorsBarrel_lastLS_<<" "<<nErrorsEndcap_lastLS_<<endl;
    double nBarrelErrors_LS = me->getBinContent(1) - nErrorsBarrel_lastLS_;
    double nEndcapErrors_LS = me->getBinContent(2) - nErrorsEndcap_lastLS_;
    double nEvents_LS = me->getBinContent(0) - nEvents_lastLS_;
    //cout<<"BINS: "<<me->getBinContent(0)<<" "<<me->getBinContent(1)<<" "<<me->getBinContent(2)<<endl;
    if(nBarrelErrors_LS/nEvents_LS>0.5) BarrelRate_LS=0.;
    if(nEndcapErrors_LS/nEvents_LS>0.5) EndcapRate_LS=0.;
    if((nBarrelErrors_LS + nEndcapErrors_LS)/nEvents_LS>0.5) PixelRate_LS=0.;
    //std::cout<<"nEvents_LS: "<<nEvents_LS<<" , nBarrelErrors_LS: "<<nBarrelErrors_LS<<" , nEndcapErrors_LS: "<<nEndcapErrors_LS<<" , BarrelRate_LS: "<<BarrelRate_LS<<" , EndcapRate_LS: "<<EndcapRate_LS<<" , PixelRate_LS: "<<PixelRate_LS<<std::endl;
  }
  
  // evaluate mean cluster charge on tracks for data quality:
  float BarrelClusterCharge = 1.;
  float EndcapClusterCharge = 1.;
  float PixelClusterCharge = 1.;
  MonitorElement * me1 = bei->get("Pixel/Clusters/OnTrack/charge_siPixelClusters_Barrel");
  if(me1 && me1->getMean()<12.) BarrelClusterCharge = 0.;
  //if(me1) cout<<"Mean cluster charge in Barrel: "<<me1->getMean()<<endl;
  MonitorElement * me2 = bei->get("Pixel/Clusters/OnTrack/charge_siPixelClusters_Endcap");
  if(me2 && me2->getMean()<12.) EndcapClusterCharge = 0.;
  //if(me2) cout<<"Mean cluster charge in Endcap: "<<me2->getMean()<<endl;
  MonitorElement * me3 = bei->get("Pixel/Clusters/OnTrack/charge_siPixelClusters");
  if(me3 && me3->getMean()<12.) PixelClusterCharge = 0.;
  //if(me3) cout<<"Mean cluster charge in Pixel: "<<me3->getMean()<<endl;
  
  // evaluate average FED occupancy for data quality:
  float BarrelOccupancy = 1.;
  float EndcapOccupancy = 1.;
  float PixelOccupancy = 1.;
  MonitorElement * me4 = bei->get("Pixel/averageDigiOccupancy");
  if(me4){
    double minBarrelOcc = 999999.; 
    double maxBarrelOcc = -1.; 
    double meanBarrelOcc = 0.;
    double minEndcapOcc = 999999.;
    double maxEndcapOcc = -1.;
    double meanEndcapOcc = 0.;
    for(int i=1; i!=41; i++){
      if(i<=32 && me4->getBinContent(i)<minBarrelOcc) minBarrelOcc=me4->getBinContent(i); 
      if(i<=32 && me4->getBinContent(i)>maxBarrelOcc) maxBarrelOcc=me4->getBinContent(i);
      if(i<=32) meanBarrelOcc+=me4->getBinContent(i);
      if(i>32 && me4->getBinContent(i)<minEndcapOcc) minEndcapOcc=me4->getBinContent(i); 
      if(i>32 && me4->getBinContent(i)>maxEndcapOcc) maxEndcapOcc=me4->getBinContent(i); 
      if(i>32) meanEndcapOcc+=me4->getBinContent(i);
      //cout<<"OCCUPANCY: "<<i<<" "<<me4->getBinContent(i)<<" : "<<minBarrelOcc<<" "<<maxBarrelOcc<<" "<<minEndcapOcc<<" "<<maxEndcapOcc<<endl;
    } 
    meanBarrelOcc = meanBarrelOcc/32.;
    meanEndcapOcc = meanEndcapOcc/8.;
    //cout<<"MEANS: "<<meanBarrelOcc<<" "<<meanEndcapOcc<<endl;
    if(minBarrelOcc<0.1*meanBarrelOcc || maxBarrelOcc>2.5*meanBarrelOcc) BarrelOccupancy=0.;
    if(minEndcapOcc<0.2*meanEndcapOcc || maxEndcapOcc>1.8*meanEndcapOcc) EndcapOccupancy=0.;
    PixelOccupancy=BarrelOccupancy*EndcapOccupancy;
    //cout<<"Occupancies: "<<meanBarrelOcc<<" "<<meanEndcapOcc<<endl;
  }
  
  float pixelFlag = PixelRate_LS * PixelClusterCharge * PixelOccupancy;
  float barrelFlag = BarrelRate_LS * BarrelClusterCharge * BarrelOccupancy;
  float endcapFlag = EndcapRate_LS * EndcapClusterCharge * EndcapOccupancy;
  //cout<<"barrel, endcap, pixel flags: "<<barrelFlag<<","<<endcapFlag<<","<<pixelFlag<<endl;
  SummaryPixel = bei->get("Pixel/EventInfo/reportSummary");
  if(SummaryPixel) SummaryPixel->Fill(pixelFlag);
  SummaryBarrel = bei->get("Pixel/EventInfo/reportSummaryContents/PixelBarrelFraction");
  if(SummaryBarrel) SummaryBarrel->Fill(barrelFlag);
  SummaryEndcap = bei->get("Pixel/EventInfo/reportSummaryContents/PixelEndcapFraction");
  if(SummaryEndcap)   SummaryEndcap->Fill(endcapFlag);
  
}

//**********************************************************************************************

void SiPixelDataQuality::fillGlobalQualityPlot(DQMStore * bei, bool init, edm::EventSetup const& eSetup, int nFEDs, bool Tier0Flag, int lumisec){
//std::cout<<"Entering SiPixelDataQuality::fillGlobalQualityPlot: "<<nFEDs<<std::endl;
  //calculate eta and phi of the modules and fill a 2D plot:
  //if(lastLS_<lumisec){ cout<<"lastLS_="<<lastLS_<<" ,lumisec="<<lumisec<<endl; lastLS_=lumisec; init=true; cout<<"init="<<init<<endl; }
  if(init){
    count=0; errcount=0;
    init=false;
    count1=0;
    count2=0;
    count3=0;
    count4=0;
    count5=0;
    count6=0;
    modCounter_=0;
  if(!Tier0Flag){
    
    //cout<<"RESETS"<<endl;
    //The plots that these Vecs are integrated throughout a run
    //So at each lumi section I save their last values (lastmods)
    //And then subtract them out later when filling the SummaryMap
    for(int j=1; j!=41; j++){
      if(allmodsVec) lastallmods_[j-1] = allmodsVec->GetBinContent(j);
      if(errmodsVec) lasterrmods_[j-1] = errmodsVec->GetBinContent(j);
      if(allmodsVec) allmodsVec->SetBinContent(j,0.);
      if(errmodsVec) errmodsVec->SetBinContent(j,0.);
      if(goodmodsVec) goodmodsVec->SetBinContent(j,0.);
    }
  }
  if(Tier0Flag){
    for(int i=1; i!=3; i++) for(int j=1; j!=8; j++){
      if(allmodsMap) allmodsMap->SetBinContent(i,j,0.);
      if(errmodsMap) errmodsMap->SetBinContent(i,j,0.);
      if(goodmodsMap) goodmodsMap->SetBinContent(i,j,0.);
    }
  }
  }
  
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
// Fill Maps:
  // Online:    
  if(nFEDs==0) return;
/*  if(!Tier0Flag){
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
    if(dname.find("Module_")!=string::npos && currDir.find("Reference")==string::npos){
      vector<string> meVec = bei->getMEs();
      int detId=-1; int fedId=-1; int linkId=-1;
      for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
        //checking for any digis or FED errors to decide if this module is in DAQ:  
        string full_path = currDir + "/" + (*it);
        //cout<<"path: "<<full_path<<" , detId= "<<detId<<endl;
        if(detId==-1 && full_path.find("SUMOFF")==string::npos &&
           (full_path.find("ndigis")!=string::npos && full_path.find("SUMDIG")==string::npos) && 
	   (getDetId(bei->get(full_path)) > 100)){
	  //cout<<"Got into the first ndigis or NErrors histogram!"<<endl;
          MonitorElement * me = bei->get(full_path);
          if (!me) continue;
	  if((full_path.find("ndigis")!=string::npos)){ 
	    modCounter_++;
            detId = getDetId(me);
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
	    int NDigis = 0;
	    if(full_path.find("ndigis")!=string::npos) NDigis = me->getEntries(); 
	    float weight = (allmodsMap->GetBinContent(fedId+1,linkId))+NDigis;
	    allmodsMap->Fill(fedId,linkId,weight);
            static const char buf[] = "Pixel/AdditionalPixelErrors/FED_%d/FedChNErrArray_%d";
            char fedplot[sizeof(buf)+4]; 
            sprintf(fedplot,buf,fedId,linkId);
	    me = bei->get(fedplot);
	    int NErrors = 0;
	    if(me) NErrors = me->getIntValue();
	    //if(fedId==37&&linkId==5) std::cout<<"THIS CHANNEL: "<<fedplot<<" , "<<NErrors<<" , "<<bei->pwd()<<std::endl;
	    if(NErrors>0) {errmodsMap->Fill(fedId,linkId,NErrors);} //if(fedId==37&&linkId==5) std::cout<<"filling errmodsMap now : "<<errmodsMap->GetBinContent(fedId+1,linkId)<<" , and: "<<NErrors<<std::endl;}
	  }
        }
      }//end loop over MEs
    }//end of module dir's
    vector<string> subDirVec = bei->getSubdirs();  
    for (vector<string>::const_iterator ic = subDirVec.begin();
         ic != subDirVec.end(); ic++) {
      bei->cd(*ic);
      init=false;
      fillGlobalQualityPlot(bei,init,eSetup,nFEDs,Tier0Flag);
      bei->goUp();
    }
    if(modCounter_==1440){
      bei->cd("Pixel/EventInfo/reportSummaryContents");
      //cout<<"B: Loop over all modules is done, now I am in    "<<bei->pwd()<<"     and currDir is    "<<currDir<<endl;
      if(bei->pwd()=="Pixel/EventInfo/reportSummaryContents"){
        SummaryReportMap = bei->get("Pixel/EventInfo/reportSummaryMap");
        if(SummaryReportMap){ 
          float contents=0.;
          for(int i=0; i!=40; i++)for(int j=1; j!=37; j++){
            //cout<<"bin: "<<i<<","<<j<<endl;
            if((allmodsMap->GetBinContent(i+1,j)) + (errmodsMap->GetBinContent(i+1,j)) > 0){
              contents = (allmodsMap->GetBinContent(i+1,j))/((allmodsMap->GetBinContent(i+1,j))+(errmodsMap->GetBinContent(i+1,j)));
            }else{
              contents = -1.;
            }
            //if(contents>=0.&&contents<0.8) std::cout<<"HERE: "<<i<<" , "<<j<<" , "<<allmodsMap->GetBinContent(i+1,j)<<" , "<<errmodsMap->GetBinContent(i+1,j)<<std::endl;
            if(i==13&&j==17&&contents>0) count1++;
            if(i==13&&j==18&&contents>0) count2++;
            if(i==15&&j==5&&contents>0) count3++;
            if(i==15&&j==6&&contents>0) count4++;
            //cout<<"\t\t MAP: "<<i<<","<<j<<","<<contents<<endl;
            if(((i==0||i==2||i==3||i==5||i==11||i==8)&&(j==1||j==8||j==13||j==17||j==20))||
               ((i==1||i==9||i==10||i==13)&&(j==1||j==5||j==8||j==20||j==22))||
               ((i==4||i==12)&&(j==5||j==10||j==13||j==17||j==22))||
               ((i==2||i==5||i==6||i==7||i==14)&&(j==5||j==10||j==22))||
               ((i==7||i==10)&&(j==13||j==17))||
               ((i==6||i==14)&&(j==1||j==20))||
               ((i==10||i==13)&&(j==10))||
               ((i==4||i==12)&&(j==2))||
               ((i==14)&&(j==15))){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+1));
            }else if(((i==16||i==19||i==21||i==24||i==26||i==27||i==29)&&(j==2||j==9||j==14||j==18||j==21))||
               ((i==17||i==18||i==25)&&(j==2||j==6||j==9||j==21||j==23))||
               ((i==20||i==23||i==28||i==31)&&(j==6||j==11||j==14||j==18||j==23))||
               ((i==21||i==22||i==26||i==29||i==30)&&(j==6||j==11||j==23))||
               ((i==18)&&(j==14||j==18))||
               ((i==22||i==30)&&(j==2||j==21))||
               ((i==18)&&(j==11))||
               ((i==17||i==25)&&(j==16))||
               ((i==19||i==27)&&(j==5))){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-1));
            }else if(i==6&&(j==14||j==15)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+2));
            }else if((i==14)&&j==14){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+3));
            }else if((i==17||i==25)&&j==17){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-3));
            }else if(((i==0||i==2||i==3||i==5||i==8||i==10||i==11||i==13)&&(j==3||j==15))||
        	     ((i==1||i==9)&&j==3)||
        	     ((i==7||i==12||i==15)&&j==15)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+4));
            }else if(((i==16||i==18||i==19||i==21||i==24||i==26||i==27||i==29)&&(j==7||j==19))||
        	     ((i==17||i==25)&&j==7)||
        	     ((i==20||i==23||i==28||i==31)&&j==19)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-4));
            }else if((i==6||i==14)&&(j==13||j==19)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+5));
            }else if((i==17||i==25)&&(j==18||j==24)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-5));
            }else if(((i==4||i==12)&&j==1)||
        	     ((i==3||i==11)&&j==6)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+6));
            }else if((i==9||i==1)&&j==4){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+8));
            }else if((i==17||i==25)&&j==12){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-8));
            }else if(((i==20||i==28)&&j==20)||
        	     ((i==19||i==27)&&j==22)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-11));
            }else if(((i==20||i==28)&&j==21)||
        	     ((i==19||i==27)&&j==23)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-13));
            }else{
              SummaryReportMap->setBinContent(i+1,j,contents);
            }
          }//end for loop over summaryReportMap bins
          for(int i=0; i!=40; i++)for(int j=1; j!=37; j++){ // catch the last few holes...
            if(((i==2||i==4||i==5||i==6||i==7||i==10||i==12||i==13||i==14||i==15)&&j==12)||
               ((i==0||i==2||i==3||i==4||i==5||i==7||i==8||i==10||i==11||i==12||i==13||i==15)&&j==24)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-8));
            }else if(((i==18||i==20||i==21||i==22||i==23||i==26||i==28||i==29||i==30||i==31)&&j==4)||
               ((i==16||i==18||i==19||i==20||i==21||i==23||i==24||i==26||i==27||i==28||i==29||i==31)&&j==16)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+8));
            }else if(((i==6||i==14)&&j==9)||
        	     ((i==3||i==11)&&j==5)||
        	     ((i==1||i==9)&&(j==11||j==16))){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-1));
            }else if(((i==17||i==25)&&j==10)||
        	     ((i==22||i==30)&&(j==8||j==15))||
        	     ((i==20||i==28)&&(j==2))){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+1));
            }else if((i==1||i==9)&&j==17){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-3));
            }else if((i==22||i==30)&&j==14){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+3));
            }else if((i==6||i==14)&&j==7){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-4));
            }else if((i==22||i==30)&&j==3){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+4));
            }else if((i==1||i==9)&&(j==18||j==24)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-5));
            }else if((i==22||i==30)&&(j==13||j==19)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+5));
            }else if(((i==20||i==28)&&j==1)||
        	     ((i==19||i==27)&&j==6)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j+6));
            }else if(((i==4||i==12)&&j==20)||
        	     ((i==3||i==11)&&j==22)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-11));
            }else if(((i==4||i==12)&&j==21)||
        	     ((i==3||i==11)&&j==23)){
              SummaryReportMap->setBinContent(i+1,j,SummaryReportMap->getBinContent(i+1,j-13));
            }
          }//end of loop over bins
          //std::cout<<"COUNTERS: "<<count1<<" , "<<count2<<" , "<<count3<<" , "<<count4<<" , "<<count5<<" , "<<count6<<std::endl;
        }//end if reportSummaryMap ME exists
      }//end if in summary directory
    }//end if modCounter_  
*/
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
  if(!Tier0Flag){
    //Not elegant, but not sure where else to put this sweet new plot!
    MonitorElement * meTmp = bei->get("Pixel/Barrel/NClustertoChargeRatio_AllMod");
    MonitorElement * meTop = bei->get("Pixel/Barrel/SUMCLU_nclusters_Barrel");
    MonitorElement * meBot = bei->get("Pixel/Barrel/SUMCLU_charge_Barrel");
    if(meTop && meBot && meTmp){
      for (int bin = 1; bin < 769; ++bin){
	float content = 0.0;
	if (meBot->getBinContent(bin) > 0.0){
	  content = meTop->getBinContent(bin)/meBot->getBinContent(bin); 
	}
	meTmp->setBinContent(bin,content);
      }
      for (int j = 0; j < 4; ++j){
	static const char buf[] = "Pixel/Barrel/NClustertoChargeRatio_NormMod%i";
	char modplot[sizeof(buf)+2];
	sprintf(modplot,buf,j+1);
	MonitorElement * meFinal = bei->get(modplot);
	if(!meFinal) continue;
	for (int i = 1; i < 769; ++i){
	  int k = 3 - j;
	  if (int(i+k)%4 == 0)
	    meFinal->setBinContent(int((i+k)/4), meTmp->getBinContent(i));
	}
	//Filling done. Now modification.
	float SFLay[3], TotLay[3];
	for (int ll = 0; ll < 3; ++ll) TotLay[ll] = 0.0;
	for (int bin = 1; bin < (meFinal->getNbinsX()+1);++bin){
          int layer     =   int((bin%48)/16);
	  TotLay[layer] += meFinal->getBinContent(bin);
	}
	float laynorm = TotLay[1]/64.;
	for (int ll = 0; ll < 3; ++ll){
	  SFLay[ll] = 0.0; if (TotLay[ll] > 0.0 && TotLay[1] > 0.0) SFLay[ll] = TotLay[1]/TotLay[ll]*(1./laynorm);
	}
	//now loop through plot
	for (int bin = 1; bin < (meFinal->getNbinsX()+1); ++bin){
	  //access the layer number for bin: int((i%48)/16)
	  int layer     =   int((bin%48)/16);
	  float content =   meFinal->getBinContent(bin);
	  //apply scale factor to bin content
	  meFinal->setBinContent(bin,content*SFLay[layer]);
	}
      }
    }
    
    eSetup.get<SiPixelFedCablingMapRcd>().get(theCablingMap);
    string currDir = bei->pwd();
    if(currDir.find("Reference")!=string::npos || currDir.find("Additional")!=string::npos) return;
    //cout<<"currDir="<<currDir<<endl;
    string dname = currDir.substr(currDir.find_last_of("/")+1);
    if(dname.find("Module_")!=string::npos && currDir.find("Reference")==string::npos){
      vector<string> meVec = bei->getMEs();
      int detId=-1; int fedId=-1;
      for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++){//loop over all modules and fill ndigis into allmodsMap
        //checking for any digis or FED errors to decide if this module is in DAQ:  
        string full_path = currDir + "/" + (*it);
        if(detId==-1 && full_path.find("SUMOFF")==string::npos &&
           (full_path.find("ndigis")!=string::npos && full_path.find("SUMDIG")==string::npos) && 
	   (getDetId(bei->get(full_path)) > 100)){
          MonitorElement * me = bei->get(full_path);
          if (!me) continue;
	  if((full_path.find("ndigis")!=string::npos)){ 
	    modCounter_++;
            detId = getDetId(me);
            for(int fedid=0; fedid!=40; ++fedid){
              SiPixelFrameConverter converter(theCablingMap.product(),fedid);
	      uint32_t newDetId = detId;
              if(converter.hasDetUnit(newDetId)){
                fedId=fedid;
                break;   
              }
            }
	    double NDigis = 0;
	    if(full_path.find("ndigis")!=string::npos) NDigis = me->getEntries(); 
	    float weight = (allmodsVec->GetBinContent(fedId+1))+NDigis;
	    //cout<<"DIGIS: "<<currDir<<" , "<<fedId<<" , "<<weight<<" , "<<NDigis<<endl;
	    allmodsVec->SetBinContent(fedId+1,weight);
	    //cout<<"\t filled: "<<allmodsVec->GetBinContent(fedId+1,weight)<<endl;
	  }
        }
      }//end loop over MEs
    }//end of module dir's
    //if(currDir.find("FED_")!=std::string::npos){
      //fillGlobalQualityPlot(bei,init,eSetup,nFEDs,Tier0Flag,lumisec);
      //bei->goUp();
    //}
    vector<string> subDirVec = bei->getSubdirs();  
    for (vector<string>::const_iterator ic = subDirVec.begin();
         ic != subDirVec.end(); ic++) {
      bei->cd(*ic);
      init=false;
      fillGlobalQualityPlot(bei,init,eSetup,nFEDs,Tier0Flag,lumisec);
      bei->goUp();
    }
    //cout<<"modCounter_: "<<modCounter_<<" , "<<bei->pwd()<<endl;
    if(modCounter_==1440){
      bei->cd("Pixel/EventInfo/reportSummaryContents");
      if(bei->pwd()=="Pixel/EventInfo/reportSummaryContents"){
        for(int i=0; i!=40; i++){//loop over FEDs to fetch the errors
          static const char buf[] = "Pixel/AdditionalPixelErrors/FED_%d/FedChNErrArray_%d";
          char fedplot[sizeof(buf)+4]; 
	  int NErrors = 0;
	  for(int j=0; j!=37; j++){//loop over FED channels within a FED
            sprintf(fedplot,buf,i,j);
	    MonitorElement * me = bei->get(fedplot);
	    if(me) NErrors = NErrors + me->getIntValue();
	  }
	  //If I fill, then I end up majorly overcounting the numbers of errors...
	  //if(NErrors>0){ errmodsVec->Fill(i,NErrors); } 
	  if(NErrors>0){ errmodsVec->SetBinContent(i+1,NErrors); } 
	}
        SummaryReportMap = bei->get("Pixel/EventInfo/reportSummaryMap");
        if(SummaryReportMap){ 
          float contents=0.;
          for(int i=1; i!=41; i++){
	    //Dynamically subtracting previous (integrated) lumi section values
	    //in order to only show current lumi section's numbers
	    float mydigis = allmodsVec->GetBinContent(i) - lastallmods_[i-1];
            float myerrs  = errmodsVec->GetBinContent(i) - lasterrmods_[i-1];
	    if ((mydigis + myerrs) > 0.){
	      contents = mydigis/(mydigis + myerrs);
	      //std::cout<<"Fed: "<<i-1<<" , nevents: "<<nevents<<" , ndigis: "<< mydigis <<" , nerrors: "<< myerrs << " , contents: " << contents << std::endl;
            }else{
	      //Changed so that dynamic zooming will still
	      //advance over these bins(in renderplugins)
              contents = -0.5;
            }
            SummaryReportMap->setBinContent(lumisec+1,i,contents);
          }//end for loop over summaryReportMap bins
        }//end if reportSummaryMap ME exists
      }//end if in summary directory
    }//end if modCounter_  
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
  }//end of offline map
  if(allmodsMap) allmodsMap->Clear();
  if(goodmodsMap) goodmodsMap->Clear();
  if(errmodsMap) errmodsMap->Clear();
}

//**********************************************************************************************

