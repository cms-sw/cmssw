#define printing false
//#define occupancyprinting false

#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelInformationExtractor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelTrackerMapCreator.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
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

#include <math.h>

#include <iostream>
using namespace std;
//=============================================================================================================
//
// -- Constructor
// 
SiPixelActionExecutor::SiPixelActionExecutor(bool offlineXMLfile, 
                                             bool Tier0Flag) : 
  offlineXMLfile_(offlineXMLfile), 
  Tier0Flag_(Tier0Flag) {
  edm::LogInfo("SiPixelActionExecutor") << 
    " Creating SiPixelActionExecutor " << "\n" ;
  configParser_ = 0;
  configWriter_ = 0;
  qtHandler_ = 0;  
  ndet_ = 0;
  //collationDone = false;
}
//=============================================================================================================
//
// --  Destructor
// 
SiPixelActionExecutor::~SiPixelActionExecutor() {
  edm::LogInfo("SiPixelActionExecutor") << 
    " Deleting SiPixelActionExecutor " << "\n" ;
  if (configParser_) delete configParser_;
  if (configWriter_) delete configWriter_;  
  if (qtHandler_) delete qtHandler_;
}
//=============================================================================================================
//
// -- Read Configuration File
//
void SiPixelActionExecutor::readConfiguration() {
  string localPath;
  if(offlineXMLfile_) localPath = string("DQM/SiPixelMonitorClient/test/sipixel_tier0_config.xml");
  else localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
  if (configParser_ == 0) {
    configParser_ = new SiPixelConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
}
//=============================================================================================================
//
// -- Read Configuration File
//
bool SiPixelActionExecutor::readConfiguration(int& tkmap_freq, 
                                              int& sum_barrel_freq, 
                                              int& sum_endcap_freq, 
					      int& sum_grandbarrel_freq, 
					      int& sum_grandendcap_freq, 
					      int& message_limit_,
					      int& source_type_,
					      int& calib_type_) {
  //printing cout<<"Entering SiPixelActionExecutor::readConfiguration..."<<endl;
  string localPath;
  if(offlineXMLfile_) localPath = string("DQM/SiPixelMonitorClient/test/sipixel_tier0_config.xml");
  else localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
  if (configParser_ == 0) {
    configParser_ = new SiPixelConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
	
  if (!configParser_->getFrequencyForTrackerMap(tkmap_freq)){
    cout << "SiPixelActionExecutor::readConfiguration: Failed to read TrackerMap configuration parameters!! ";
    return false;
  }
  if (!configParser_->getFrequencyForBarrelSummary(sum_barrel_freq)){
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read Barrel Summary configuration parameters!! " << "\n" ;
    return false;
  }
  if (!configParser_->getFrequencyForEndcapSummary(sum_endcap_freq)){
    edm::LogInfo("SiPixelActionExecutor")  << "Failed to read Endcap Summary configuration parameters!! " << "\n" ;
    return false;
  }
  if (!configParser_->getFrequencyForGrandBarrelSummary(sum_grandbarrel_freq)){
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read Grand Barrel Summary configuration parameters!! " << "\n" ;
    return false;
  }
  if (!configParser_->getFrequencyForGrandEndcapSummary(sum_grandendcap_freq)){
    edm::LogInfo("SiPixelActionExecutor")  << "Failed to read Grand Endcap Summary configuration parameters!! " << "\n" ;
    return false;
  }
  if (!configParser_->getMessageLimitForQTests(message_limit_)){
    edm::LogInfo("SiPixelActionExecutor")  << "Failed to read QTest Message Limit" << "\n" ;
    return false;
  }
  if (!configParser_->getSourceType(source_type_)){
    edm::LogInfo("SiPixelActionExecutor")  << "Failed to read Source Type" << "\n" ;
    return false;
  }
  if (!configParser_->getCalibType(calib_type_)){
    edm::LogInfo("SiPixelActionExecutor")  << "Failed to read Calib Type" << "\n" ;
    return false;
  }
  //printing cout<<"...leaving SiPixelActionExecutor::readConfiguration..."<<endl;
  return true;
}
//=============================================================================================================
bool SiPixelActionExecutor::readConfiguration(int& tkmap_freq, int& summary_freq) {
  //printing cout<<"Entering SiPixelActionExecutor::readConfiguration..."<<endl;
  string localPath;
  if(offlineXMLfile_) localPath = string("DQM/SiPixelMonitorClient/test/sipixel_tier0_config.xml");
  else localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
  if (configParser_ == 0) {
    configParser_ = new SiPixelConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
	
  if (!configParser_->getFrequencyForTrackerMap(tkmap_freq)){
    cout << "SiPixelActionExecutor::readConfiguration: Failed to read TrackerMap configuration parameters!! ";
    return false;
  }
  if (!configParser_->getFrequencyForBarrelSummary(summary_freq)){
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read Summary configuration parameters!! " << "\n" ;
    return false;
  }
  //printing cout<<"...leaving SiPixelActionExecutor::readConfiguration..."<<endl;
  return true;
}
//=============================================================================================================
// -- Create Tracker Map
//
void SiPixelActionExecutor::createTkMap(DQMStore* bei, 
                                        string mEName,
					string theTKType) 
{
	
  SiPixelTrackerMapCreator tkmap_creator(mEName,theTKType,offlineXMLfile_);
  tkmap_creator.create(bei);
	
/*     cout << ACYellow << ACBold 
  	  << "[SiPixelActionExecutor::createTkMap()]"
  	  << ACPlain
  	  << " Tracker map created (type:" 
  	  << theTKType
  	  << ")"
  	  << endl;
*/
}

//=============================================================================================================
void SiPixelActionExecutor::createSummary(DQMStore* bei) {
  //printing cout<<"entering SiPixelActionExecutor::createSummary..."<<endl;
  string barrel_structure_name;
  vector<string> barrel_me_names;
  string localPath;
  if(offlineXMLfile_) localPath = string("DQM/SiPixelMonitorClient/test/sipixel_tier0_config.xml");
  else localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
  //cout<<"*********************ATTENTION! LOCALPATH= "<<localPath<<endl;
  if (configParser_ == 0) {
    configParser_ = new SiPixelConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
  if (!configParser_->getMENamesForBarrelSummary(barrel_structure_name, barrel_me_names)){
    cout << "SiPixelActionExecutor::createSummary: Failed to read Barrel Summary configuration parameters!! ";
    return;
  }
  configParser_->getSourceType(source_type_); 
  //cout<<"++++++++++++++++++++++++++SOURCE TYPE= "<<source_type_<<endl;
  bei->setCurrentFolder("Pixel/");
  //bei->cd();
  fillSummary(bei, barrel_structure_name, barrel_me_names, true); // Barrel
  bei->setCurrentFolder("Pixel/");
  //bei->cd();
  string endcap_structure_name;
  vector<string> endcap_me_names;
  if (!configParser_->getMENamesForEndcapSummary(endcap_structure_name, endcap_me_names)){
    edm::LogInfo("SiPixelActionExecutor")  << "Failed to read Endcap Summary configuration parameters!! " << "\n" ;
    return;
  }

  // printing cout << "--- Processing endcap" << endl;

  bei->setCurrentFolder("Pixel/");
  //bei->cd();
  fillSummary(bei, endcap_structure_name, endcap_me_names, false); // Endcap
  bei->setCurrentFolder("Pixel/");
  //bei->cd();
  if(source_type_==0||source_type_==5 || source_type_ == 20){//do this only if RawData source is present
    string federror_structure_name;
    vector<string> federror_me_names;
    if (!configParser_->getMENamesForFEDErrorSummary(federror_structure_name, federror_me_names)){
      cout << "SiPixelActionExecutor::createSummary: Failed to read FED Error Summary configuration parameters!! ";
      return;
    }
    bei->setCurrentFolder("Pixel/");
    //bei->cd();
    fillFEDErrorSummary(bei, federror_structure_name, federror_me_names);
    bei->setCurrentFolder("Pixel/");
    //bei->cd();
  }
  //createLayout(bei);
  //string fname = "test.xml";
  // configWriter_->write(fname);
  if (configWriter_) delete configWriter_;
  configWriter_ = 0;
  //printing cout<<"leaving SiPixelActionExecutor::createSummary..."<<endl;
}

//=============================================================================================================

void SiPixelActionExecutor::GetBladeSubdirs(DQMStore* bei, vector<string>& blade_subdirs) {
	blade_subdirs.clear();
//	cout << "BladeSubdirs::" << bei->pwd() << endl;
	vector<string> panels = bei->getSubdirs();
	vector<string> modules;
	for (vector<string>::const_iterator it = panels.begin(); it != panels.end(); it++) {
		bei->cd(*it);
		modules = bei->getSubdirs();
		for (vector<string>::const_iterator m_it = modules.begin(); m_it != modules.end(); m_it++) {
//			cout << "Would have added " << (*m_it) << "." << endl;
			blade_subdirs.push_back(*m_it);
		}
	}
//	cout << "Got Blade subdirs" << endl;
}


//=============================================================================================================

void SiPixelActionExecutor::fillSummary(DQMStore* bei, string dir_name, vector<string>& me_names, bool isbarrel)
{
	

  //printing cout<<"entering SiPixelActionExecutor::fillSummary..."<<endl;
  string currDir = bei->pwd();
  string prefix;
  if(source_type_==0) prefix="SUMRAW";
  else if (source_type_==1) prefix="SUMDIG";
  else if (source_type_==2) prefix="SUMCLU";
  else if (source_type_==3) prefix="SUMTRK";
  else if (source_type_==4) prefix="SUMHIT";
  else if (source_type_>=7 && source_type_<20) prefix="SUMCAL";
  else if (source_type_==20) prefix="SUMOFF";
  if (currDir.find(dir_name) != string::npos)  {
    vector<MonitorElement*> sum_mes;
    for (vector<string>::const_iterator iv = me_names.begin();
	 iv != me_names.end(); iv++) {
      if(source_type_==5||source_type_==6){
	if((*iv)=="errorType"||(*iv)=="NErrors"||(*iv)=="fullType"||(*iv)=="chanNmbr"||
	   (*iv)=="TBMType"||(*iv)=="EvtNbr"||(*iv)=="evtSize"||(*iv)=="linkId"||
	   (*iv)=="ROCId"||(*iv)=="DCOLId"||(*iv)=="PXId"||(*iv)=="ROCNmbr"||
	   (*iv)=="TBMMessage"||(*iv)=="Type36Hitmap") 
	  prefix="SUMRAW";
	else if((*iv)=="ndigis"||(*iv)=="adc")
	  prefix="SUMDIG";
	else if((*iv)=="nclusters"||(*iv)=="x"||(*iv)=="y"||(*iv)=="charge"||
		(*iv)=="size"||(*iv)=="sizeX"||(*iv)=="sizeY"||(*iv)=="minrow"||
		(*iv)=="maxrow"||(*iv)=="mincol"||(*iv)=="maxcol")
	  prefix="SUMCLU";
          if(currDir.find("Track")!=string::npos) prefix="SUMTRK";
	else if((*iv)=="residualX"||(*iv)=="residualY")
	  prefix="SUMTRK";
	else if((*iv)=="ClustX"||(*iv)=="ClustY"||(*iv)=="nRecHits"||(*iv)=="ErrorX"||(*iv)=="ErrorY")
	  prefix="SUMHIT";
	else if((*iv)=="Gain1d"||(*iv)=="GainChi2NDF1d"||
		(*iv)=="GainChi2Prob1d"||(*iv)=="Pedestal1d"||
		(*iv)=="GainNPoints1d"||(*iv)=="GainHighPoint1d"||
		(*iv)=="GainLowPoint1d"||(*iv)=="GainEndPoint1d"||
		(*iv)=="GainFitResult2d"||(*iv)=="GainDynamicRange2d"||
		(*iv)=="GainSaturate2d"||
		(*iv)=="ScurveChi2ProbSummary"||(*iv)=="ScurveFitResultSummary"||
		(*iv)=="ScurveSigmasSummary"||(*iv)=="ScurveThresholdSummary"||
		(*iv)=="pixelAliveSummary"  || (*iv) == "SiPixelErrorsCalibDigis") 
	  prefix="SUMCAL"; 
      }
      MonitorElement* temp; string tag;
      if((*iv).find("residual")!=string::npos){                           // track residuals
	tag = prefix + "_" + (*iv) + "_mean_" 
	  + currDir.substr(currDir.find(dir_name));
	temp = getSummaryME(bei, tag);
	sum_mes.push_back(temp);
	tag = prefix + "_" + (*iv) + "_RMS_" 
	  + currDir.substr(currDir.find(dir_name));
	temp = getSummaryME(bei, tag);
	sum_mes.push_back(temp);
      }else if(prefix == "SUMCAL"){                  // calibrations
	if((*iv)=="Gain1d" || (*iv)=="GainChi2NDF1d" || (*iv)=="GainChi2Prob1d" ||
	   (*iv)=="GainNPoints1d" || (*iv)=="GainHighPoint1d" ||
	   (*iv)=="GainLowPoint1d" || (*iv)=="GainEndPoint1d" || 
	   (*iv)=="GainDynamicRange2d" || (*iv)=="GainSaturate2d" ||
	   (*iv)=="Pedestal1d" ||
	   (*iv)=="ScurveChi2ProbSummary" || (*iv)=="ScurveFitResultSummary" ||
	   (*iv)=="ScurveSigmasSummary" || (*iv)=="ScurveThresholdSummary"){                    
	  tag = prefix + "_" + (*iv) + "_mean_" 
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag);
	  sum_mes.push_back(temp);
	  tag = prefix + "_" + (*iv) + "_RMS_" 
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag);
	  sum_mes.push_back(temp);
	}else if((*iv) == "SiPixelErrorsCalibDigis"){
	  tag = prefix + "_" + (*iv) + "_NCalibErrors_"
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag);
	  sum_mes.push_back(temp);
	}else if((*iv)=="GainFitResult2d"){
	  tag = prefix + "_" + (*iv) + "_NNegativeFits_"
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag);
	  sum_mes.push_back(temp);
	}else if((*iv)=="pixelAliveSummary"){
	  tag = prefix + "_" + (*iv) + "_FracOfPerfectPix_"
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag);
	  sum_mes.push_back(temp);
	  tag = prefix + "_" + (*iv) + "_mean_"
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag);
	  sum_mes.push_back(temp);
	}
      }else{
	tag = prefix + "_" + (*iv) + "_" + currDir.substr(currDir.find(dir_name));
	temp = getSummaryME(bei, tag);
	sum_mes.push_back(temp);
	if((*iv)=="ndigis"){
	  tag = prefix + "_" + (*iv) + "FREQ_" 
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag);
	  sum_mes.push_back(temp);
	}
	if(prefix=="SUMDIG" && (*iv)=="adc"){
	  tag = "ALLMODS_" + (*iv) + "COMB_" + currDir.substr(currDir.find(dir_name));
	  temp = bei->book1D(tag.c_str(), tag.c_str(),256, 0., 256.);
	  sum_mes.push_back(temp);
	}
	if(prefix=="SUMCLU" && (*iv)=="charge"){
	  tag = "ALLMODS_" + (*iv) + "COMB_" + currDir.substr(currDir.find(dir_name));
	  temp = bei->book1D(tag.c_str(), tag.c_str(),500, 0., 500.); // To look to get the size automatically	  
	  sum_mes.push_back(temp);
	}
      }
    }
    if (sum_mes.size() == 0) {
      edm::LogInfo("SiPixelActionExecutor") << " Summary MEs can not be created" << "\n" ;
      return;
    }
    vector<string> subdirs = bei->getSubdirs();
//Blade
  if(dir_name.find("Blade_") == 0) GetBladeSubdirs(bei, subdirs);
	
    int ndet = 0;
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      if (prefix!="SUMOFF" && (*it).find("Module_") == string::npos) continue;
      if (prefix=="SUMOFF" && (*it).find(isbarrel?"Layer_":"Disk_") == string::npos) continue;
      bei->cd(*it);
      ndet++;

      vector<string> contents = bei->getMEs(); 
			
			for (vector<MonitorElement*>::const_iterator isum = sum_mes.begin();
				 isum != sum_mes.end(); isum++) {
				for (vector<string>::const_iterator im = contents.begin();
					 im != contents.end(); im++) {
					string sname = ((*isum)->getName());
					string tname = " ";
					tname = sname.substr(7,(sname.find("_",7)-6));
					if(sname.find("ALLMODS_adcCOMB_")!=string::npos) tname = "adc_";
					if(sname.find("ALLMODS_chargeCOMB_")!=string::npos) tname = "charge_";
					if(sname.find("_charge_")!=string::npos && sname.find("Track_")==string::npos) tname = "charge_";
					if(sname.find("_nclusters_")!=string::npos && sname.find("Track_")==string::npos) tname = "nclusters_";
					if(sname.find("_size_")!=string::npos && sname.find("Track_")==string::npos) tname = "size_";
					if(sname.find("_charge_OffTrack_")!=string::npos) tname = "charge_OffTrack_";
					if(sname.find("_nclusters_OffTrack_")!=string::npos) tname = "nclusters_OffTrack_";
					if(sname.find("_size_OffTrack_")!=string::npos) tname = "size_OffTrack_";
					if(sname.find("_sizeX_OffTrack_")!=string::npos) tname = "sizeX_OffTrack_";
					if(sname.find("_sizeY_OffTrack_")!=string::npos) tname = "sizeY_OffTrack_";
					if(sname.find("_charge_OnTrack_")!=string::npos) tname = "charge_OnTrack_";
					if(sname.find("_nclusters_OnTrack_")!=string::npos) tname = "nclusters_OnTrack_";
					if(sname.find("_size_OnTrack_")!=string::npos) tname = "size_OnTrack_";
					if(sname.find("_sizeX_OnTrack_")!=string::npos) tname = "sizeX_OnTrack_";
					if(sname.find("_sizeY_OnTrack_")!=string::npos) tname = "sizeY_OnTrack_";
					//if(sname.find("ALLMODS")!=string::npos) cout<<"sname and tname= "<<sname<<","<<tname<<endl;
					if(tname.find("FREQ")!=string::npos) tname = "ndigis_";
					if (((*im)).find(tname) == 0) {
						string fullpathname = bei->pwd() + "/" + (*im); 
					//cout<<"!!!!!!!!!!!!!!!!!!!!!!SNAME= "<<sname<<endl;	
	    MonitorElement *  me = bei->get(fullpathname);

	    if (me){ 
	      if (sname.find("_RMS_")!=string::npos && 
		  sname.find("GainDynamicRange2d")==string::npos && 
		  sname.find("GainSaturate2d")==string::npos){
		(*isum)->Fill(ndet, me->getRMS());
	      }else if (sname.find("GainDynamicRange2d")!=string::npos ||
			sname.find("GainSaturate2d")!=string::npos){
		float SumOfEntries=0.; float SumOfSquaredEntries=0.; int SumOfPixels=0;
		for(int cols=1; cols!=me->getNbinsX()+1; cols++) for(int rows=1; rows!=me->getNbinsY()+1; rows++){
		  SumOfEntries+=me->getBinContent(cols,rows);
		  SumOfSquaredEntries+=(me->getBinContent(cols,rows))*(me->getBinContent(cols,rows));
		  SumOfPixels++;
		}

		float MeanInZ = SumOfEntries / float(SumOfPixels);
		float RMSInZ = sqrt(SumOfSquaredEntries/float(SumOfPixels));
		if(sname.find("_mean_")!=string::npos) (*isum)->Fill(ndet, MeanInZ);
		if(sname.find("_RMS_")!=string::npos) (*isum)->Fill(ndet, RMSInZ);
	      }else if (sname.find("_FracOfPerfectPix_")!=string::npos){
		//printing cout<<"nbins = "<<me->getNbinsX()<<" , "<<me->getBinContent(me->getNbinsX()-1)<<" , "<<me->getBinContent(me->getNbinsX())<<endl;
		float nlast = me->getBinContent(me->getNbinsX());
		float nall = (me->getTH1F())->Integral(1,11);
		//printing cout << nall << endl;
		(*isum)->Fill(ndet, nlast/nall);
	      }else if (sname.find("_NCalibErrors_")!=string::npos ||
			sname.find("FREQ_")!=string::npos){
		float nall = me->getEntries();
		(*isum)->Fill(ndet, nall);
	      }else if (sname.find("GainFitResult2d")!=string::npos){
		int NegFitPixels=0;
		for(int cols=1; cols!=me->getNbinsX()+1; cols++) for(int rows=1; rows!=me->getNbinsY()+1; rows++){
		  if(me->getBinContent(cols,rows)<0.) NegFitPixels++;
		}
		(*isum)->Fill(ndet, float(NegFitPixels));
	      }else if (sname.find("ALLMODS_adcCOMB_")!=string::npos || sname.find("ALLMODS_chargeCOMB_")!=string::npos){
		(*isum)->getTH1F()->Add(me->getTH1F());
	      }else if (sname.find("_NErrors_")!=string::npos){
	        string path1 = fullpathname;
		path1 = path1.replace(path1.find("NErrors"),7,"errorType");
		MonitorElement * me1 = bei->get(path1);
		bool othererror=false;
	        if(me1){
	          for(int jj=1; jj<16; jj++){
	            if(me1->getBinContent(jj)>0.){
	              if(jj==6){ //errorType=30 (reset)
	                string path2 = path1;
			path2 = path2.replace(path2.find("errorType"),9,"TBMMessage");
	                MonitorElement * me2 = bei->get(path2);
	                if(me2) for(int kk=1; kk<9; kk++) if(me2->getBinContent(kk)>0.) if(kk!=6 && kk!=7) 
			  othererror=true;
		      }else{ //not reset, but other error
		        othererror=true;
		      }
		    }
		  }
		}
		if(othererror) (*isum)->Fill(ndet, me->getMean());
	      }else{
		(*isum)->Fill(ndet, me->getMean());
	      }
	      if(prefix=="SUMOFF"){
		(*isum)->setAxisTitle(isbarrel?"Ladders":"Blades",1);
	      }else if(sname.find("ALLMODS_adcCOMB_")!=string::npos){
		(*isum)->setAxisTitle("Digi charge [ADC]",1);
	      }else if(sname.find("ALLMODS_chargeCOMB_")!=string::npos){
		(*isum)->setAxisTitle("Cluster charge [kilo electrons]",1);
	      }else{
		(*isum)->setAxisTitle("Modules",1);
	      }
	      string title = " ";
	      if (sname.find("_RMS_")!=string::npos){
		title = "RMS of " + sname.substr(7,(sname.find("_",7)-7)) + " per module"; 
	      }else if (sname.find("_FracOfPerfectPix_")!=string::npos){
		title = "FracOfPerfectPix " + sname.substr(7,(sname.find("_",7)-7)) + " per module"; 
	      }else if(sname.find("_NCalibErrors_")!=string::npos){
		title = "Number of CalibErrors " + sname.substr(7,(sname.find("_",7)-7)) + " per module"; 
	      }else if(sname.find("_NNegativeFits_")!=string::npos){
		title = "Number of pixels with neg. fit result " + sname.substr(7,(sname.find("_",7)-7)) + " per module"; 
	      }else if (sname.find("FREQ_")!=string::npos){
		title = "NEvents with digis per module"; 
	      }else if (sname.find("ALLMODS_adcCOMB_")!=string::npos){
		title = "NDigis";
	      }else if (sname.find("ALLMODS_chargeCOMB_")!=string::npos){
		title = "NClusters";
	      }else{
		if(prefix=="SUMOFF") title = "Mean " + sname.substr(7,(sname.find("_",7)-7)) + (isbarrel?" per Ladder":"per Blade"); 
		else title = "Mean " + sname.substr(7,(sname.find("_",7)-7)) + " per Module"; 
	      }
	      (*isum)->setAxisTitle(title,2);
	    }
	    break;
	  }
	}
      }
      bei->goUp();
	  if(dir_name.find("Blade") == 0) bei->goUp(); // Going up a second time if we are processing the Blade
    } // end for it (subdirs)
  } else {  
    vector<string> subdirs = bei->getSubdirs();
    // printing cout << "#\t" << bei->pwd() << endl;
    if(isbarrel)
      {
			
	for (vector<string>::const_iterator it = subdirs.begin();
	     it != subdirs.end(); it++) {
	  //				 cout << "##\t" << bei->pwd() << "\t" << (*it) << endl;
	  if((bei->pwd()).find("Endcap")!=string::npos ||
	     (bei->pwd()).find("AdditionalPixelErrors")!=string::npos) bei->goUp();
	  bei->cd(*it);
	  if((*it).find("Endcap")!=string::npos ||
	     (*it).find("AdditionalPixelErrors")!=string::npos) continue;
	  fillSummary(bei, dir_name, me_names, true); // Barrel
	  bei->goUp();
	}
	string grandbarrel_structure_name;
	vector<string> grandbarrel_me_names;
	if (!configParser_->getMENamesForGrandBarrelSummary(grandbarrel_structure_name, grandbarrel_me_names)){
	  cout << "SiPixelActionExecutor::createSummary: Failed to read Grand Barrel Summary configuration parameters!! ";
	  return;
	}
	fillGrandBarrelSummaryHistos(bei, grandbarrel_me_names);
			
      }
    else // Endcap
      {
			
	for (vector<string>::const_iterator it = subdirs.begin();
	     it != subdirs.end(); it++) {
	  //				 cout << "##\t" << bei->pwd() << "\t" << (*it) << endl;
	  if((bei->pwd()).find("Barrel")!=string::npos ||
	     (bei->pwd()).find("AdditionalPixelErrors")!=string::npos) bei->goUp();
	  bei->cd((*it));
	  if ((*it).find("Barrel")!=string::npos ||
	      (*it).find("AdditionalPixelErrors")!=string::npos) continue;
	  fillSummary(bei, dir_name, me_names, false); // Endcap
	  bei->goUp();
	}
	string grandendcap_structure_name;
	vector<string> grandendcap_me_names;
	if (!configParser_->getMENamesForGrandEndcapSummary(grandendcap_structure_name, grandendcap_me_names)){
	  cout << "SiPixelActionExecutor::createSummary: Failed to read Grand Endcap Summary configuration parameters!! ";
	  return;
	}
	fillGrandEndcapSummaryHistos(bei, grandendcap_me_names);
			
			
      }
  }
  //printing cout<<"...leaving SiPixelActionExecutor::fillSummary!"<<endl;
	
  // End of cleanup
	
	
}

//=============================================================================================================
void SiPixelActionExecutor::fillFEDErrorSummary(DQMStore* bei,
                                                string dir_name,
						vector<string>& me_names) {
  //printing cout<<"entering SiPixelActionExecutor::fillFEDErrorSummary..."<<endl;
  string currDir = bei->pwd();
  string prefix;
  if(source_type_==0) prefix="SUMRAW";
  else if(source_type_==20) prefix="SUMOFF";
  
  if (currDir.find(dir_name) != string::npos)  {
    vector<MonitorElement*> sum_mes;
    for (vector<string>::const_iterator iv = me_names.begin();
	 iv != me_names.end(); iv++) {
      if(source_type_==5||source_type_==6){
	if((*iv)=="errorType"||(*iv)=="NErrors"||(*iv)=="fullType"||(*iv)=="chanNmbr"||
	   (*iv)=="TBMType"||(*iv)=="EvtNbr"||(*iv)=="evtSize"||(*iv)=="linkId"||
	   (*iv)=="ROCId"||(*iv)=="DCOLId"||(*iv)=="PXId"||(*iv)=="ROCNmbr"||
	   (*iv)=="TBMMessage"||(*iv)=="Type36Hitmap") 
	  prefix="SUMRAW";
      }
      string tag = prefix + "_" + (*iv) + "_FEDErrors";
      MonitorElement* temp = getFEDSummaryME(bei, tag);
      sum_mes.push_back(temp);
    }
    if (sum_mes.size() == 0) {
      edm::LogInfo("SiPixelActionExecutor") << " Summary MEs can not be created" << "\n" ;
      return;
    }
    vector<string> subdirs = bei->getSubdirs();
    int ndet = 0;
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      if ( (*it).find("FED_") == string::npos) continue;
      bei->cd(*it);
      ///////      ndet++;
      string fedid = (*it).substr((*it).find("_")+1);
      if(fedid=="0") ndet = 1;
      else if(fedid=="1") ndet = 2;
      else if(fedid=="2") ndet = 3;
      else if(fedid=="3") ndet = 4;
      else if(fedid=="4") ndet = 5;
      else if(fedid=="5") ndet = 6;
      else if(fedid=="6") ndet = 7;
      else if(fedid=="7") ndet = 8;
      else if(fedid=="8") ndet = 9;
      else if(fedid=="9") ndet = 10;
      else if(fedid=="10") ndet = 11;
      else if(fedid=="11") ndet = 12;
      else if(fedid=="12") ndet = 13;
      else if(fedid=="13") ndet = 14;
      else if(fedid=="14") ndet = 15;
      else if(fedid=="15") ndet = 16;
      else if(fedid=="16") ndet = 17;
      else if(fedid=="17") ndet = 18;
      else if(fedid=="18") ndet = 19;
      else if(fedid=="19") ndet = 20;
      else if(fedid=="20") ndet = 21;
      else if(fedid=="21") ndet = 22;
      else if(fedid=="22") ndet = 23;
      else if(fedid=="23") ndet = 24;
      else if(fedid=="24") ndet = 25;
      else if(fedid=="25") ndet = 26;
      else if(fedid=="26") ndet = 27;
      else if(fedid=="27") ndet = 28;
      else if(fedid=="28") ndet = 29;
      else if(fedid=="29") ndet = 30;
      else if(fedid=="30") ndet = 31;
      else if(fedid=="31") ndet = 32;
      else if(fedid=="32") ndet = 33;
      else if(fedid=="33") ndet = 34;
      else if(fedid=="34") ndet = 35;
      else if(fedid=="35") ndet = 36;
      else if(fedid=="36") ndet = 37;
      else if(fedid=="37") ndet = 38;
      else if(fedid=="38") ndet = 39;
      else if(fedid=="39") ndet = 40;
      vector<string> contents = bei->getMEs(); 
			
      for (vector<MonitorElement*>::const_iterator isum = sum_mes.begin();
	   isum != sum_mes.end(); isum++) {
	for (vector<string>::const_iterator im = contents.begin();
	     im != contents.end(); im++) {
	  string sname = ((*isum)->getName());
	  string tname = " ";
	  tname = sname.substr(7,(sname.find("_",7)-6));
	  if (((*im)).find(tname) == 0) {
	    string fullpathname = bei->pwd() + "/" + (*im); 
	    MonitorElement *  me = bei->get(fullpathname);
						
	    if (me){ 
	      if(me->getMean()>0.){
	        if (sname.find("_NErrors_")!=string::npos){
	          string path1 = fullpathname;
		  path1 = path1.replace(path1.find("NErrors"),7,"errorType");
		  MonitorElement * me1 = bei->get(path1);
		  bool othererror=false;
	          if(me1){
	            for(int jj=1; jj<16; jj++){
	              if(me1->getBinContent(jj)>0.){
	                if(jj==6){ //errorType=30 (reset)
	                  string path2 = path1;
			  path2 = path2.replace(path2.find("errorType"),9,"TBMMessage");
	                  MonitorElement * me2 = bei->get(path2);
	                  if(me2) for(int kk=1; kk<9; kk++) if(me2->getBinContent(kk)>0.) if(kk!=6 && kk!=7) 
			    othererror=true;
		        }else{ //not reset, but other error
		          othererror=true;
		        }
		      }
		    }
		  }
		  if(othererror) (*isum)->Fill(ndet, me->getMean());
	        }else (*isum)->Fill(ndet-1, me->getMean());
	      }
	      (*isum)->setAxisTitle("FED #",1);
	      string title = " ";
	      title = "Mean " + sname.substr(7,(sname.find("_",7)-7)) + " per FED"; 
	      (*isum)->setAxisTitle(title,2);
	    }
	    break;
	  }
	}
      }
      bei->goUp();
    }
  } else {  
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      if((*it).find("Endcap")!=string::npos ||
	 (*it).find("Barrel")!=string::npos) continue;
      bei->cd(*it);
      fillFEDErrorSummary(bei, dir_name, me_names);
      bei->goUp();
    }
  }
  //printing cout<<"...leaving SiPixelActionExecutor::fillFEDErrorSummary!"<<endl;
}


//=============================================================================================================
void SiPixelActionExecutor::fillGrandBarrelSummaryHistos(DQMStore* bei,
                                                         vector<string>& me_names) {
//  cout<<"Entering SiPixelActionExecutor::fillGrandBarrelSummaryHistos...:"<<me_names.size()<<endl;
  vector<MonitorElement*> gsum_mes;
  string currDir = bei->pwd();
  string path_name = bei->pwd();
  string dir_name =  path_name.substr(path_name.find_last_of("/")+1);
//  cout<<"I am in "<<path_name<<" now."<<endl;
  if ((dir_name.find("DQMData") == 0) ||
      (dir_name.find("Pixel") == 0) ||
      (dir_name.find("AdditionalPixelErrors") == 0) ||
      (dir_name.find("Endcap") == 0) ||
      (dir_name.find("HalfCylinder") == 0) ||
      (dir_name.find("Disk") == 0) ||
      (dir_name.find("Blade") == 0) ||
      (dir_name.find("Panel") == 0) ) return;
  vector<string> subdirs = bei->getSubdirs();
  int nDirs = subdirs.size();
  int iDir =0;
  int nbin = 0;
  int nbin_i = 0; 
  int nbin_subdir = 0; 
  int cnt=0;
  bool first_subdir = true;
  for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
    cnt++;
    bei->cd(*it);
//    cout << "--- " << cnt << "\t" << bei->pwd() << endl;
    vector<string> contents = bei->getMEs();
		
    bei->goUp();
		
		string prefix;
		if(source_type_==0) prefix="SUMRAW";
		else if (source_type_==1) prefix="SUMDIG";
		else if (source_type_==2) prefix="SUMCLU";
		else if (source_type_==3) prefix="SUMTRK";
		else if (source_type_==4) prefix="SUMHIT";
		else if (source_type_>=7 && source_type_<20) prefix="SUMCAL";
		else if (source_type_==20) prefix="SUMOFF";

		
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
//      cout<<"A: iterating over "<<(*im)<<" now:"<<endl;
      for (vector<string>::const_iterator iv = me_names.begin();
	   iv != me_names.end(); iv++) {
	string var = "_" + (*iv) + "_";
//	cout<<"\t B: iterating over "<<(*iv)<<" now, var is set to: "<<var<<endl;
	if ((*im).find(var) != string::npos) {
	  if((var=="_charge_" || var=="_nclusters_" || var=="_size_" || var=="_sizeX_" || var=="_sizeY_") && 
	     (*im).find("Track_")!=string::npos) continue;
//	  cout << "Looking into " << (*iv) << endl;
	  string full_path = (*it) + "/" +(*im);
	  MonitorElement * me = bei->get(full_path.c_str());
	  if (!me) continue; 
	  if(source_type_==5||source_type_==6){
	    if((*iv)=="errorType"||(*iv)=="NErrors"||(*iv)=="fullType"||(*iv)=="chanNmbr"||
	       (*iv)=="TBMType"||(*iv)=="EvtNbr"||(*iv)=="evtSize"||(*iv)=="linkId"||
	       (*iv)=="ROCId"||(*iv)=="DCOLId"||(*iv)=="PXId"||(*iv)=="ROCNmbr"||
	       (*iv)=="TBMMessage"||(*iv)=="Type36Hitmap") 
	      prefix="SUMRAW";
	    else if((*iv)=="ndigis"||(*iv)=="adc" ||
		    (*iv)=="ndigisFREQ" || (*iv)=="adcCOMB")
	      prefix="SUMDIG";
	    else if((*iv)=="nclusters"||(*iv)=="x"||(*iv)=="y"||(*iv)=="charge"||(*iv)=="chargeCOMB"||
		    (*iv)=="size"||(*iv)=="sizeX"||(*iv)=="sizeY"||(*iv)=="minrow"||
		    (*iv)=="maxrow"||(*iv)=="mincol"||(*iv)=="maxcol")
	      prefix="SUMCLU";
	      if(currDir.find("Track")!=string::npos) prefix="SUMTRK";
	    else if((*iv)=="residualX_mean"||(*iv)=="residualY_mean"||
		    (*iv)=="residualX_RMS"||(*iv)=="residualY_RMS")
	      prefix="SUMTRK";
	    else if((*iv)=="ClustX"||(*iv)=="ClustY"||(*iv)=="nRecHits"||(*iv)=="ErrorX"||(*iv)=="ErrorY")
	      prefix="SUMHIT";
	    else if((*iv)=="Gain1d_mean"||(*iv)=="GainChi2NDF1d_mean"||
		    (*iv)=="GainChi2Prob1d_mean"||(*iv)=="Pedestal1d_mean"||
		    (*iv)=="ScurveChi2ProbSummary_mean"||(*iv)=="ScurveFitResultSummary_mean"||
		    (*iv)=="ScurveSigmasSummary_mean"||(*iv)=="ScurveThresholdSummary_mean"||
		    (*iv)=="Gain1d_RMS"||(*iv)=="GainChi2NDF1d_RMS"||
		    (*iv)=="GainChi2Prob1d_RMS"||(*iv)=="Pedestal1d_RMS"||
		    (*iv)=="GainNPoints1d_mean" || (*iv)=="GainNPoints1d_RMS" ||
		    (*iv)=="GainHighPoint1d_mean" || (*iv)=="GainHighPoint1d_RMS" ||
		    (*iv)=="GainLowPoint1d_mean" || (*iv)=="GainLowPoint1d_RMS" ||
		    (*iv)=="GainEndPoint1d_mean" || (*iv)=="GainEndPoint1d_RMS" ||
		    (*iv)=="GainFitResult2d_mean" || (*iv)=="GainFitResult2d_RMS" ||
		    (*iv)=="GainDynamicRange2d_mean" || (*iv)=="GainDynamicRange2d_RMS" ||
		    (*iv)=="GainSaturate2d_mean" || (*iv)=="GainSaturate2d_RMS" ||
		    (*iv)=="ScurveChi2ProbSummary_RMS"||(*iv)=="ScurveFitResultSummary_RMS"||
		    (*iv)=="ScurveSigmasSummary_RMS"||(*iv)=="ScurveThresholdSummary_RMS"||
		    (*iv)=="pixelAliveSummary_mean"||(*iv)=="pixelAliveSummary_FracOfPerfectPix" ||
		    (*iv)=="SiPixelErrorsCalibDigis_NCalibErrors" )
	      prefix="SUMCAL";
	  } // end source_type if
	  
	  // bugfix: gsum_mes is filled using the contents of me_names. Proceeding with each entry in me_names.
      /*
	  int actual_size = gsum_mes.size();
	  int wanted_size = me_names.size();
	  // printing cout << actual_size << "\t" << wanted_size << endl;
	  if (actual_size !=  wanted_size) { */
	  if (first_subdir){
//	    bool create_me = true;
	    nbin = me->getTH1F()->GetNbinsX();        
	    string me_name = prefix + "_" + (*iv) + "_" + dir_name;
	    if((*iv)=="adcCOMB"||(*iv)=="chargeCOMB") me_name = "ALLMODS_" + (*iv) + "_" + dir_name;
	    else if(prefix=="SUMOFF" && dir_name=="Barrel") nbin=192;
	    else if((*iv)=="adcCOMB") nbin=256;
	    else if(dir_name=="Barrel") nbin=768;
	    else if(prefix=="SUMOFF" && dir_name.find("Shell")!=string::npos) nbin=48;
	    else if(dir_name.find("Shell")!=string::npos) nbin=192;
	    else nbin=nbin*nDirs;

		getGrandSummaryME(bei, nbin, me_name, gsum_mes);
	  }
	  /*
	    for (vector<MonitorElement*>::const_iterator igm = gsum_mes.begin();
		 igm !=gsum_mes.end(); igm++) { 
	      // To be further optimized
	      if( (*iv).find("Clust") != string::npos && (*igm)->getName().find(me_name) != string::npos ) create_me = false; //cout << "Already have it" << endl;
	    }
	  
	    // printing cout<<"me_name to be created= "<<me_name<<endl;
	    if(create_me) getGrandSummaryME(bei, nbin, me_name, gsum_mes);
	  }
	  */
	  // end bugfix: gsum_mes.


	  for (vector<MonitorElement*>::const_iterator igm = gsum_mes.begin();
	       igm != gsum_mes.end(); igm++) {
//	    cout<<"\t \t C: iterating over "<<(*igm)->getName()<<" now:"<<endl;
	    if ((*igm)->getName().find(var) != string::npos) {
//	      cout<<"\t \t D: Have the correct var now!"<<endl;
	      if(prefix=="SUMOFF") (*igm)->setAxisTitle("Ladders",1);
	      else if((*igm)->getName().find("adcCOMB_")!=string::npos) (*igm)->setAxisTitle("Digi charge [ADC]",1);
	      else if((*igm)->getName().find("chargeCOMB_")!=string::npos) (*igm)->setAxisTitle("Cluster charge [kilo electrons]",1);
	      else (*igm)->setAxisTitle("Modules",1);

	      // Setting title

		  string title="";
	      if(prefix=="SUMOFF") title = "mean " + (*iv) + " per Ladder"; 
	      else if((*igm)->getName().find("FREQ_") != string::npos) title = "NEvents with digis per Module"; 
	      else if((*igm)->getName().find("adcCOMB_") != string::npos) title = "NDigis";
	      else if((*igm)->getName().find("chargeCOMB_") != string::npos) title = "NClusters";
	      else title = "mean " + (*iv) + " per Module"; 
	      (*igm)->setAxisTitle(title,2);
		  
		  // Setting binning

	      if((*igm)->getName().find("ALLMODS_adcCOMB_")!=string::npos){
		nbin_subdir=256;
	      }else if((*igm)->getName().find("ALLMODS_chargeCOMB_")!=string::npos){
		nbin_subdir=500;
	      }else if((*igm)->getName().find("Ladder") != string::npos){
		nbin_i=0; nbin_subdir=4;
	      }else if((*igm)->getName().find("Layer") != string::npos){
		nbin_i=(cnt-1)*4; nbin_subdir=4;
	      }else if((*igm)->getName().find("Shell") != string::npos){
		if(prefix!="SUMOFF"){
		  if(iDir==0){ nbin_i=0; nbin_subdir=40; }
		  else if(iDir==1){ nbin_i=40; nbin_subdir=64; }
		  else if(iDir==2){ nbin_i=104; nbin_subdir=88; }
		}else{
		  if(iDir==0){ nbin_i=0; nbin_subdir=10; }
		  else if(iDir==1){ nbin_i=10; nbin_subdir=16; }
		  else if(iDir==2){ nbin_i=26; nbin_subdir=22; }
		}
	      }else if((*igm)->getName().find("Barrel") != string::npos){
		if(prefix!="SUMOFF"){
		  if(iDir==0){ nbin_i=0; nbin_subdir=192; }
		  else if(iDir==1){ nbin_i=192; nbin_subdir=192; }
		  else if(iDir==2){ nbin_i=384; nbin_subdir=192; }
		  else if(iDir==3){ nbin_i=576; nbin_subdir=192; }
		}else{
		  if(iDir==0){ nbin_i=0; nbin_subdir=48; }
		  else if(iDir==1){ nbin_i=48; nbin_subdir=48; }
		  else if(iDir==2){ nbin_i=96; nbin_subdir=48; }
		  else if(iDir==3){ nbin_i=144; nbin_subdir=48; }
		}
	      }


			if((*igm)->getName().find("ndigisFREQ")==string::npos)
			{ 
				if(((*igm)->getName().find("adcCOMB")!=string::npos && me->getName().find("adcCOMB")!=string::npos) 
						|| ((*igm)->getName().find("chargeCOMB")!=string::npos && me->getName().find("chargeCOMB")!=string::npos))
				{
					(*igm)->getTH1F()->Add(me->getTH1F());
				}else if(((*igm)->getName().find("charge_")!=string::npos && (*igm)->getName().find("Track_")==string::npos && 
				          me->getName().find("charge_")!=string::npos && me->getName().find("Track_")==string::npos) || 
					 ((*igm)->getName().find("nclusters_")!=string::npos && (*igm)->getName().find("Track_")==string::npos && 
				          me->getName().find("nclusters_")!=string::npos && me->getName().find("Track_")==string::npos) || 
					 ((*igm)->getName().find("size_")!=string::npos && (*igm)->getName().find("Track_")==string::npos && 
				          me->getName().find("size_")!=string::npos && me->getName().find("Track_")==string::npos) || 
					 ((*igm)->getName().find("charge_OffTrack_")!=string::npos && me->getName().find("charge_OffTrack_")!=string::npos) || 
					 ((*igm)->getName().find("nclusters_OffTrack_")!=string::npos && me->getName().find("nclusters_OffTrack_")!=string::npos) || 
					 ((*igm)->getName().find("size_OffTrack_")!=string::npos && me->getName().find("size_OffTrack_")!=string::npos) || 
					 ((*igm)->getName().find("charge_OnTrack_")!=string::npos && me->getName().find("charge_OnTrack_")!=string::npos) || 
					 ((*igm)->getName().find("nclusters_OnTrack_")!=string::npos && me->getName().find("nclusters_OnTrack_")!=string::npos) || 
					 ((*igm)->getName().find("size_OnTrack_")!=string::npos && me->getName().find("size_OnTrack_")!=string::npos) || 
				         ((*igm)->getName().find("charge_")==string::npos && (*igm)->getName().find("nclusters_")==string::npos && 
					  (*igm)->getName().find("size_")==string::npos)){
					for (int k = 1; k < nbin_subdir+1; k++) if(me->getBinContent(k) > 0) (*igm)->setBinContent(k+nbin_i, me->getBinContent(k));
				}
			}
			else if(me->getName().find("ndigisFREQ")!=string::npos)
			{
				for (int k = 1; k < nbin_subdir+1; k++) if(me->getBinContent(k) > 0) (*igm)->setBinContent(k+nbin_i, me->getBinContent(k));
			}
	    } // end var in igm (gsum_mes)
	  } // end igm loop
	} // end var in im (contents)
      } // end of iv loop
    } // end of im loop
    iDir++;
  first_subdir = false; // We are done processing the first directory, we don't add any new MEs in the future passes.	
  } // end of it loop (subdirs)
//  cout<<"...leaving SiPixelActionExecutor::fillGrandBarrelSummaryHistos!"<<endl;
}

//=============================================================================================================
void SiPixelActionExecutor::fillGrandEndcapSummaryHistos(DQMStore* bei,
                                                         vector<string>& me_names) {
  //printing cout<<"Entering SiPixelActionExecutor::fillGrandEndcapSummaryHistos..."<<endl;
  vector<MonitorElement*> gsum_mes;
  string currDir = bei->pwd();
  string path_name = bei->pwd();
  string dir_name =  path_name.substr(path_name.find_last_of("/")+1);
  if ((dir_name.find("DQMData") == 0) ||
      (dir_name.find("Pixel") == 0) ||
      (dir_name.find("AdditionalPixelErrors") == 0) ||
      (dir_name.find("Barrel") == 0) ||
      (dir_name.find("Shell") == 0) ||
      (dir_name.find("Layer") == 0) ||
      (dir_name.find("Ladder") == 0) ) return;
  vector<string> subdirs = bei->getSubdirs();
  int iDir =0;
  int nbin = 0;
  int nbin_i = 0; 
  int nbin_subdir = 0; 
  int cnt=0;
  bool first_subdir = true;  
  for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
    cnt++;
    bei->cd(*it);
    vector<string> contents = bei->getMEs();
    bei->goUp();
		
    string prefix;
    if(source_type_==0) prefix="SUMRAW";
    else if (source_type_==1) prefix="SUMDIG";
    else if (source_type_==2) prefix="SUMCLU";
    else if (source_type_==3) prefix="SUMTRK";
    else if (source_type_==4) prefix="SUMHIT";
    else if (source_type_>=7 && source_type_<20) prefix="SUMCAL";
    else if (source_type_==20) prefix="SUMOFF";
		
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
      for (vector<string>::const_iterator iv = me_names.begin();
	   iv != me_names.end(); iv++) {
	string var = "_" + (*iv) + "_";
	if ((*im).find(var) != string::npos) {
	  if((var=="_charge_" || var=="_nclusters_" || var=="_size_" || var=="_sizeX_" || var=="_sizeY_") && 
	     (*im).find("Track_")!=string::npos) continue;
	  string full_path = (*it) + "/" +(*im);
	  MonitorElement * me = bei->get(full_path.c_str());
	  if (!me) continue; 
	  if(source_type_==5||source_type_==6){
	    if((*iv)=="errorType"||(*iv)=="NErrors"||(*iv)=="fullType"||(*iv)=="chanNmbr"||
	       (*iv)=="TBMType"||(*iv)=="EvtNbr"||(*iv)=="evtSize"||(*iv)=="linkId"||
	       (*iv)=="ROCId"||(*iv)=="DCOLId"||(*iv)=="PXId"||(*iv)=="ROCNmbr"||
	       (*iv)=="TBMMessage"||(*iv)=="Type36Hitmap") 
	      prefix="SUMRAW";
	    else if((*iv)=="ndigis"||(*iv)=="adc" ||
		    (*iv)=="ndigisFREQ"||(*iv)=="adcCOMB")
	      prefix="SUMDIG";
	    else if((*iv)=="nclusters"||(*iv)=="x"||(*iv)=="y"||(*iv)=="charge"||(*iv)=="chargeCOMB"||
		    (*iv)=="size"||(*iv)=="sizeX"||(*iv)=="sizeY"||(*iv)=="minrow"||
		    (*iv)=="maxrow"||(*iv)=="mincol"||(*iv)=="maxcol")
	      prefix="SUMCLU";
	      if(currDir.find("Track")!=string::npos) prefix="SUMTRK";
	    else if((*iv)=="residualX_mean"||(*iv)=="residualY_mean"||
		    (*iv)=="residualX_RMS"||(*iv)=="residualY_RMS")
	      prefix="SUMTRK";
	    else if((*iv)=="ClustX"||(*iv)=="ClustY"||(*iv)=="nRecHits"||(*iv)=="ErrorX"||(*iv)=="ErrorY")
	      prefix="SUMHIT";
	    else if((*iv)=="Gain1d_mean"||(*iv)=="GainChi2NDF1d_mean"||
		    (*iv)=="GainChi2Prob1d_mean"||(*iv)=="Pedestal1d_mean"||
		    (*iv)=="ScurveChi2ProbSummary_mean"||(*iv)=="ScurveFitResultSummary_mean"||
		    (*iv)=="ScurveSigmasSummary_mean"||(*iv)=="ScurveThresholdSummary_mean"||
		    (*iv)=="Gain1d_RMS"||(*iv)=="GainChi2NDF1d_RMS"||
		    (*iv)=="GainChi2Prob1d_RMS"||(*iv)=="Pedestal1d_RMS"||
		    (*iv)=="GainNPoints1d_mean" || (*iv)=="GainNPoints1d_RMS" ||
		    (*iv)=="GainHighPoint1d_mean" || (*iv)=="GainHighPoint1d_RMS" ||
		    (*iv)=="GainLowPoint1d_mean" || (*iv)=="GainLowPoint1d_RMS" ||
		    (*iv)=="GainEndPoint1d_mean" || (*iv)=="GainEndPoint1d_RMS" ||
		    (*iv)=="GainFitResult2d_mean" || (*iv)=="GainFitResult2d_RMS" ||
		    (*iv)=="GainDynamicRange2d_mean" || (*iv)=="GainDynamicRange2d_RMS" ||
		    (*iv)=="GainSaturate2d_mean" || (*iv)=="GainSaturate2d_RMS" ||
		    (*iv)=="ScurveChi2ProbSummary_RMS"||(*iv)=="ScurveFitResultSummary_RMS"||
		    (*iv)=="ScurveSigmasSummary_RMS"||(*iv)=="ScurveThresholdSummary_RMS"||
		    (*iv)=="pixelAliveSummary_mean"||(*iv)=="pixelAliveSummary_FracOfPerfectPix"|| 
		    (*iv) == "SiPixelErrorsCalibDigis_NCalibErrors")
	      prefix="SUMCAL"; 
	  }

	  // bugfix: gsum_mes is filled using the contents of me_names. Proceeding with each entry in me_names.
	  /*
	  int actual_size = gsum_mes.size();
	  int wanted_size = me_names.size();
	  if (actual_size !=  wanted_size) { */
	  if (first_subdir){
//	    bool create_me = true;
	    nbin = me->getTH1F()->GetNbinsX();        
	    string me_name = prefix + "_" + (*iv) + "_" + dir_name;
	    if((*iv)=="adcCOMB"||(*iv)=="chargeCOMB") me_name = "ALLMODS_" + (*iv) + "_" + dir_name;
	    else if(prefix=="SUMOFF" && dir_name=="Endcap") nbin=96;
	    else if(dir_name=="Endcap") nbin=672;
	    else if(prefix=="SUMOFF" && dir_name.find("HalfCylinder")!=string::npos) nbin=24;
	    else if(dir_name.find("HalfCylinder")!=string::npos) nbin=168;
	    else if(prefix=="SUMOFF" && dir_name.find("Disk")!=string::npos) nbin=12;
	    else if(dir_name.find("Disk")!=string::npos) nbin=84;
	    else if(dir_name.find("Blade")!=string::npos) nbin=7;
	    //else if(dir_name.find("Panel_1")!=string::npos) nbin=4;
	    //else if(dir_name.find("Panel_2")!=string::npos) nbin=3;
		//cout << dir_name.c_str() << "\t" << nbin << endl;
		getGrandSummaryME(bei, nbin, me_name, gsum_mes);
	  }
	  /*

	    for (vector<MonitorElement*>::const_iterator igm = gsum_mes.begin();
		 igm !=gsum_mes.end(); igm++) { 
	      // To be further optimized
	      if( (*iv).find("Clust") != string::npos && (*igm)->getName().find(me_name) != string::npos ) create_me = false; //cout << "Already have it" << endl;
	    }

	    // printing cout<<"me_name to be created= "<<me_name<<endl;
	    if(create_me) getGrandSummaryME(bei, nbin, me_name, gsum_mes);
	  }
	  */
	  // end bugfix: gsum_mes.

	  for (vector<MonitorElement*>::const_iterator igm = gsum_mes.begin();
	       igm != gsum_mes.end(); igm++) { 
	    if ((*igm)->getName().find(var) != string::npos) {
	      if(prefix=="SUMOFF") (*igm)->setAxisTitle("Blades",1);
	      else if((*igm)->getName().find("adcCOMB_")!=string::npos) (*igm)->setAxisTitle("Digi charge [ADC]",1);
	      else if((*igm)->getName().find("chargeCOMB_")!=string::npos) (*igm)->setAxisTitle("Cluster charge [kilo electrons]",1);
	      else (*igm)->setAxisTitle("Modules",1);
	      string title="";
	      if(prefix=="SUMOFF") title = "mean " + (*iv) + " per Blade"; 
	      else if((*igm)->getName().find("FREQ_") != string::npos) title = "NEvents with digis per Module"; 
	      else if((*igm)->getName().find("adcCOMB_")!=string::npos) title = "NDigis";
	      else if((*igm)->getName().find("chargeCOMB_")!=string::npos) title = "NClusters";
	      else title = "mean " + (*iv) + " per Module"; 
	      (*igm)->setAxisTitle(title,2);
	      nbin_i=0; 
	      if((*igm)->getName().find("ALLMODS_adcCOMB_")!=string::npos){
		nbin_subdir=256;
	      }else if((*igm)->getName().find("ALLMODS_chargeCOMB_")!=string::npos){
		nbin_subdir=500;
	      }else if((*igm)->getName().find("Panel_") != string::npos){
		nbin_subdir=7;
//	      }else if((*igm)->getName().find("Panel_1") != string::npos){
//		nbin_subdir=4;
//	      }else if((*igm)->getName().find("Panel_2") != string::npos){
//		nbin_subdir=3;
	      }else if((*igm)->getName().find("Blade") != string::npos){
		if((*im).find("_1") != string::npos) nbin_subdir=4;
		if((*im).find("_2") != string::npos) {nbin_i=4; nbin_subdir=3;}
	      }else if((*igm)->getName().find("Disk") != string::npos){
		nbin_i=((cnt-1)%12)*7; nbin_subdir=7;
	      }else if((*igm)->getName().find("HalfCylinder") != string::npos){
		if(prefix!="SUMOFF"){
		  nbin_subdir=84;
		  if((*im).find("_2") != string::npos) nbin_i=84;
		}else{
		  nbin_subdir=12;
		  if((*im).find("_2") != string::npos) nbin_i=12;
		}
	      }else if((*igm)->getName().find("Endcap") != string::npos){
		if(prefix!="SUMOFF"){
		  nbin_subdir=168;
		  if((*im).find("_mO") != string::npos) nbin_i=168;
		  if((*im).find("_pI") != string::npos) nbin_i=336;
		  if((*im).find("_pO") != string::npos) nbin_i=504;
		}else{
		  nbin_subdir=24;
		  if((*im).find("_mO") != string::npos) nbin_i=24;
		  if((*im).find("_pI") != string::npos) nbin_i=48;
		  if((*im).find("_pO") != string::npos) nbin_i=72;
		}
	      }
							
	      //	       for (int k = 1; k < nbin_subdir+1; k++) {
	      if((*igm)->getName().find("ndigisFREQ")==string::npos){ 
		if(((*igm)->getName().find("adcCOMB")!=string::npos && me->getName().find("adcCOMB")!=string::npos) || ((*igm)->getName().find("chargeCOMB")!=string::npos && me->getName().find("chargeCOMB")!=string::npos)){
		  (*igm)->getTH1F()->Add(me->getTH1F());
		}else if(((*igm)->getName().find("charge_")!=string::npos && (*igm)->getName().find("Track_")==string::npos && 
			   me->getName().find("charge_")!=string::npos && me->getName().find("Track_")==string::npos) || 
			  ((*igm)->getName().find("nclusters_")!=string::npos && (*igm)->getName().find("Track_")==string::npos && 
			   me->getName().find("nclusters_")!=string::npos && me->getName().find("Track_")==string::npos) || 
			  ((*igm)->getName().find("size_")!=string::npos && (*igm)->getName().find("Track_")==string::npos && 
			   me->getName().find("size_")!=string::npos && me->getName().find("Track_")==string::npos) || 
			  ((*igm)->getName().find("charge_OffTrack_")!=string::npos && me->getName().find("charge_OffTrack_")!=string::npos) || 
			  ((*igm)->getName().find("nclusters_OffTrack_")!=string::npos && me->getName().find("nclusters_OffTrack_")!=string::npos) || 
			  ((*igm)->getName().find("size_OffTrack_")!=string::npos && me->getName().find("size_OffTrack_")!=string::npos) || 
			  ((*igm)->getName().find("charge_OnTrack_")!=string::npos && me->getName().find("charge_OnTrack_")!=string::npos) || 
			  ((*igm)->getName().find("nclusters_OnTrack_")!=string::npos && me->getName().find("nclusters_OnTrack_")!=string::npos) || 
			  ((*igm)->getName().find("size_OnTrack_")!=string::npos && me->getName().find("size_OnTrack_")!=string::npos) || 
			  ((*igm)->getName().find("charge_")==string::npos && (*igm)->getName().find("nclusters_")==string::npos && 
			   (*igm)->getName().find("size_")==string::npos)){
		  for (int k = 1; k < nbin_subdir+1; k++) if(me->getBinContent(k) > 0) (*igm)->setBinContent(k+nbin_i, me->getBinContent(k));
		}
	      }else if(me->getName().find("ndigisFREQ")!=string::npos){
		for (int k = 1; k < nbin_subdir+1; k++)  if(me->getBinContent(k) > 0) (*igm)->setBinContent(k+nbin_i, me->getBinContent(k));
	      }
	      //	       }// for
							
	    }
	  }
	}
      }
    }
		
    iDir++;
  first_subdir = false; // We are done processing the first directory, we don't add any new MEs in the future passes.	
  }  // end for it (subdirs)
}
//=============================================================================================================
//
// -- Get Summary ME
//
void SiPixelActionExecutor::getGrandSummaryME(DQMStore* bei,
                                              int nbin, 
					      string& me_name, 
					      vector<MonitorElement*> & mes) {
  //printing cout<<"Entering SiPixelActionExecutor::getGrandSummaryME for: "<<me_name<<endl;
  if((bei->pwd()).find("Pixel")==string::npos) return; // If one doesn't find pixel
  vector<string> contents = bei->getMEs();
	
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    //printing cout<<"in grand summary me: "<<me_name<<","<<(*it)<<endl;
    if ((*it).find(me_name) == 0) {
      string fullpathname = bei->pwd() + "/" + me_name;
      //				cout << "###\t" << fullpathname << endl;
      MonitorElement* me = bei->get(fullpathname);
			
      if (me) {
	//      cout<<"Found grand ME: "<<fullpathname<<endl;
	me->Reset();
	mes.push_back(me);
	// if printing cout<<"reset and add the following me: "<<me->getName()<<endl;
	return;
      }
    }
  }

  //  MonitorElement* temp_me = bei->book1D(me_name.c_str(),me_name.c_str(),nbin,1.,nbin+1.);
  //  if (temp_me) mes.push_back(temp_me);
  MonitorElement* temp_me(0);
  if(me_name.find("ALLMODS_adcCOMB_")!=string::npos || me_name.find("ALLMODS_chargeCOMB_")!=string::npos) temp_me = bei->book1D(me_name.c_str(),me_name.c_str(),nbin,0,nbin);
  else temp_me = bei->book1D(me_name.c_str(),me_name.c_str(),nbin,1.,nbin+1.);
  if (temp_me) mes.push_back(temp_me);
	
  //  if(temp_me) cout<<"finally found grand ME: "<<me_name<<endl;
}


//=============================================================================================================
//
// -- Get Summary ME
//
MonitorElement* SiPixelActionExecutor::getSummaryME(DQMStore* bei,
                                                    string me_name) {
  //printing cout<<"Entering SiPixelActionExecutor::getSummaryME for: "<<me_name<<endl;
  MonitorElement* me = 0;
  if((bei->pwd()).find("Pixel")==string::npos) return me;
  vector<string> contents = bei->getMEs();    
	
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(me_name) == 0) {
      string fullpathname = bei->pwd() + "/" + (*it); 
      me = bei->get(fullpathname);
			
      if (me) {
	//printing cout<<"got this ME: "<<fullpathname<<endl;
	me->Reset();
	return me;
      }
    }
  }
  contents.clear();
//	cout << me_name.c_str() 
//		<< "\t" << ((me_name.find("SUMOFF")==string::npos)?"true":"false")
//		<< "\t" << ((me_name.find("Blade_")!= string::npos)?"true":"false")
//		<< "\t" << ((me_name.find("Layer1_")!=string::npos)?"true":"false")
//		<< "\t" << ((me_name.find("Layer2_")!=string::npos)?"true":"false")
//		<< "\t" << ((me_name.find("Layer3_")!=string::npos)?"true":"false")
//		<< "\t" << ((me_name.find("Disk_")!=string::npos)?"true":"false")
//		<< endl;
  if(me_name.find("SUMOFF")==string::npos){
  	if(me_name.find("Blade_")!=string::npos)me = bei->book1D(me_name.c_str(), me_name.c_str(),7,1.,8.);
	else me = bei->book1D(me_name.c_str(), me_name.c_str(),4,1.,5.);
//    if(me_name.find("Panel_2")!=string::npos)  me = bei->book1D(me_name.c_str(), me_name.c_str(),3,1.,4.);
//    else me = bei->book1D(me_name.c_str(), me_name.c_str(),4,1.,5.);
  }else if(me_name.find("Layer_1")!=string::npos){ me = bei->book1D(me_name.c_str(), me_name.c_str(),10,1.,11.);
  }else if(me_name.find("Layer_2")!=string::npos){ me = bei->book1D(me_name.c_str(), me_name.c_str(),16,1.,17.);
  }else if(me_name.find("Layer_3")!=string::npos){ me = bei->book1D(me_name.c_str(), me_name.c_str(),22,1.,23.);
  }else if(me_name.find("Disk_")!=string::npos){ me = bei->book1D(me_name.c_str(), me_name.c_str(),12,1.,13.);
  }
	
  //  if(me) cout<<"Finally got this ME: "<<me_name<<endl;
  //if(me_name.find("ALLMODS_adc_")!=string::npos) me = bei->book1D(me_name.c_str(), me_name.c_str(),256, 0., 256.);
	
  //printing cout<<"...leaving SiPixelActionExecutor::getSummaryME!"<<endl;
  return me;
}


//=============================================================================================================
MonitorElement* SiPixelActionExecutor::getFEDSummaryME(DQMStore* bei,
                                                       string me_name) {
  //printing cout<<"Entering SiPixelActionExecutor::getFEDSummaryME..."<<endl;
  MonitorElement* me = 0;
  if((bei->pwd()).find("Pixel")==string::npos) return me;
  vector<string> contents = bei->getMEs();
	
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(me_name) == 0) {
      string fullpathname = bei->pwd() + "/" + (*it); 
			
      me = bei->get(fullpathname);
			
      if (me) {
	//printing cout<<"got the ME: "<<fullpathname<<endl;
	me->Reset();
	return me;
      }
    }
  }
  contents.clear();
  me = bei->book1D(me_name.c_str(), me_name.c_str(),40,-0.5,39.5);
  //if(me) cout<<"finally got the ME: "<<me_name<<endl;
  return me;
  //printing cout<<"...leaving SiPixelActionExecutor::getFEDSummaryME!"<<endl;
}

//=============================================================================================================
void SiPixelActionExecutor::bookOccupancyPlots(DQMStore* bei, bool hiRes, bool isbarrel) // Polymorphism
{
  if(Tier0Flag_) return;
  vector<string> subdirs = bei->getSubdirs();
  for (vector<string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++)
    {
      if(isbarrel && (*it).find("Barrel")==string::npos) continue;
      if(!isbarrel && (*it).find("Endcap")==string::npos) continue;
		
      if((*it).find("Module_")!=string::npos) continue;
      if((*it).find("Panel_")!=string::npos) continue;
      bei->cd(*it);
      bookOccupancyPlots(bei, hiRes, isbarrel);
      if(!hiRes){
	//occupancyprinting cout<<"booking low res barrel occ plot now!"<<endl;
	OccupancyMap = bei->book2D((isbarrel?"barrelOccupancyMap":"endcapOccupancyMap"),"Barrel Digi Occupancy Map (4 pix per bin)",isbarrel?208:130,0.,isbarrel?416.:260.,80,0.,160.);
      }else{
	//occupancyprinting cout<<"booking high res barrel occ plot now!"<<endl;
	OccupancyMap = bei->book2D((isbarrel?"barrelOccupancyMap":"endcapOccupancyMap"),"Barrel Digi Occupancy Map (1 pix per bin)",isbarrel?416:260,0.,isbarrel?416.:260.,160,0.,160.);
      }
      OccupancyMap->setAxisTitle("Columns",1);
      OccupancyMap->setAxisTitle("Rows",2);
		
      bei->goUp();
		
    }
	
	
	
}
//=============================================================================================================
void SiPixelActionExecutor::bookOccupancyPlots(DQMStore* bei, bool hiRes) {
	
  if(Tier0Flag_) return;
  // Barrel
  bei->cd();
  bei->setCurrentFolder("Pixel");
  this->bookOccupancyPlots(bei, hiRes, true);
	
  // Endcap
  bei->cd();
  bei->setCurrentFolder("Pixel");
  this->bookOccupancyPlots(bei, hiRes, false);
	
}

void SiPixelActionExecutor::createOccupancy(DQMStore* bei) {
  //std::cout<<"entering SiPixelActionExecutor::createOccupancy..."<<std::endl;
  if(Tier0Flag_) return;
  bei->cd();
  fillOccupancy(bei, true);
  bei->cd();
  fillOccupancy(bei, false);
  bei->cd();
  //std::cout<<"leaving SiPixelActionExecutor::createOccupancy..."<<std::endl;
}

//=============================================================================================================

void SiPixelActionExecutor::fillOccupancy(DQMStore* bei, bool isbarrel)
{
  //occupancyprinting cout<<"entering SiPixelActionExecutor::fillOccupancy..."<<std::endl;
  if(Tier0Flag_) return;
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  //occupancyprinting cout<<"currDir= "<<currDir<< " , dname= "<<dname<<std::endl;
	
  if(dname.find("Module_")!=string::npos && currDir.find("Pixel/Endcap/HalfCylinder_mI/Disk_1/Blade_01/Panel_2/Module_2")==string::npos){ // Skipping noisy module/ROC
    vector<string> meVec = bei->getMEs();
    for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
      string full_path = currDir + "/" + (*it);
      if(full_path.find("hitmap_siPixelDigis")!=string::npos){ // If we have the hitmap ME
	MonitorElement * me = bei->get(full_path);
	if (!me) continue;
	//occupancyprinting cout << full_path << endl;
	string path = full_path;
	while (path.find_last_of("/") != 5) // Stop before Pixel/
	  {
	    path = path.substr(0,path.find_last_of("/"));
	    //							cout << "\t" << path << endl;
	    OccupancyMap = bei->get(path + "/" + (isbarrel?"barrel":"endcap") + "OccupancyMap");
					
	    if(OccupancyMap){ 
	      //occupancyprinting cout<<"I found the occupancy map!"<<std::endl;
						
						
	      if(!isbarrel)
		{
							
		  //		cout << full_path << endl;
		  //		
		  //		cout << "OccupancyMap" <<endl;
		  //		cout << "X:\t" << OccupancyMap->getTH2F()->GetNbinsX() << "\tY:\t" << OccupancyMap->getTH2F()->GetNbinsY() << endl;
		  //		cout << OccupancyMap->getTH2F()->ProjectionX()->GetBinLowEdge(1) << "\t" << OccupancyMap->getTH2F()->ProjectionX()->GetBinLowEdge(OccupancyMap->getTH2F()->GetNbinsX()+1) << endl;
		  //		cout << OccupancyMap->getTH2F()->ProjectionY()->GetBinLowEdge(1) << "\t" << OccupancyMap->getTH2F()->ProjectionY()->GetBinLowEdge(OccupancyMap->getTH2F()->GetNbinsY()+1) << endl;
		  //
		  //		cout << "ME" << endl;
		  //		cout << "X:\t" << me->getTH2F()->GetNbinsX() << "\tY:\t" << me->getTH2F()->GetNbinsY() << endl;
		  //		cout << me->getTH2F()->ProjectionX()->GetBinLowEdge(1) << "\t" << me->getTH2F()->ProjectionX()->GetBinLowEdge(me->getTH2F()->GetNbinsX()+1) << endl;
		  //		cout << me->getTH2F()->ProjectionY()->GetBinLowEdge(1) << "\t" << me->getTH2F()->ProjectionY()->GetBinLowEdge(me->getTH2F()->GetNbinsY()+1) << endl;
		  //		cout << "--------------------" << endl;						
							
		}
						
	      if(isbarrel && full_path.find("F/")!=string::npos) OccupancyMap->getTH2F()->Add(me->getTH2F());
	      if(!isbarrel || (isbarrel && full_path.find("H/")!=string::npos)) 
		{  
		  TH2F *tmpHist = (TH2F*) OccupancyMap->getTH2F()->Clone("tmpHist");
		  tmpHist->Reset();
							
		  for(int i=1; i!=me->getNbinsX()+1; i++) for(int j=1; j!=me->getNbinsY()+1; j++) tmpHist->SetBinContent(i,j,me->getBinContent(i,j));
		  //							me->getTH2F()->Print();
		  OccupancyMap->getTH2F()->Add(tmpHist);
							
		  tmpHist->Delete();
		}
						
	      OccupancyMap->getTH2F()->SetEntries(OccupancyMap->getTH2F()->Integral());
						
	    }       
					
	  }
      }
			
    }
    //bei->goUp();
  } else {  
    //occupancyprinting cout<<"finding subdirs now"<<std::endl;
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++) {
      bei->cd(*it);
      //occupancyprinting cout<<"now I am in "<<bei->pwd()<<std::endl;
      if(*it != "Pixel" && ((isbarrel && (*it).find("Barrel")==string::npos) || (!isbarrel && (*it).find("Endcap")==string::npos))) continue;
      //occupancyprinting cout<<"calling myself again "<<std::endl;
      fillOccupancy(bei, isbarrel);
      bei->goUp();
    }
  }
	
  //occupancyprinting cout<<"leaving SiPixelActionExecutor::fillOccupancy..."<<std::endl;
	
}
//=============================================================================================================

//
// -- Setup Quality Tests 
//
void SiPixelActionExecutor::setupQTests(DQMStore * bei) {
  //printing cout<<"Entering SiPixelActionExecutor::setupQTests: "<<endl;
	
  bei->cd();
  bei->cd("Pixel");
	
  string localPath;
  if(offlineXMLfile_) localPath = string("DQM/SiPixelMonitorClient/test/sipixel_tier0_qualitytest.xml");
  else localPath = string("DQM/SiPixelMonitorClient/test/sipixel_qualitytest_config.xml");
  if(!qtHandler_){
    qtHandler_ = new QTestHandle();
  }
  if(!qtHandler_->configureTests(edm::FileInPath(localPath).fullPath(),bei)){
    qtHandler_->attachTests(bei,false);
    bei->cd();
  }else{
    cout << " Problem setting up quality tests "<<endl;
  }
	
  //printing cout<<" leaving SiPixelActionExecutor::setupQTests. "<<endl;
}
//=============================================================================================================
//
// -- Check Status of Quality Tests
//
void SiPixelActionExecutor::checkQTestResults(DQMStore * bei) {
  //printing cout<<"Entering SiPixelActionExecutor::checkQTestResults..."<<endl;
	
  int messageCounter=0;
  string currDir = bei->pwd();
  vector<string> contentVec;
  bei->getContents(contentVec);
  configParser_->getCalibType(calib_type_);
  //  cout << calib_type_ << endl;
  configParser_->getMessageLimitForQTests(message_limit_);
  for (vector<string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {
    vector<string> contents;
    int nval = SiPixelUtility::getMEList((*it), contents);
    if (nval == 0) continue;
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
			
      MonitorElement * me = bei->get((*im));
      if (me) {
	me->runQTests();
	// get all warnings associated with me
	vector<QReport*> warnings = me->getQWarnings();
	for(vector<QReport *>::const_iterator wi = warnings.begin();
	    wi != warnings.end(); ++wi) {
	  messageCounter++;
	  if(messageCounter<message_limit_) {
	    //edm::LogWarning("SiPixelQualityTester::checkTestResults") << 
	    //  " *** Warning for " << me->getName() << 
	    //  "," << (*wi)->getMessage() << "\n";
						
	    edm::LogWarning("SiPixelActionExecutor::checkQTestResults") <<  " *** Warning for " << me->getName() << "," 
									<< (*wi)->getMessage() << " " << me->getMean() 
									<< " " << me->getRMS() << me->hasWarning() 
									<< endl;
	  }
	}
	warnings=vector<QReport*>();
	// get all errors associated with me
	vector<QReport *> errors = me->getQErrors();
	for(vector<QReport *>::const_iterator ei = errors.begin();
	    ei != errors.end(); ++ei) {
					
	  float empty_mean = me->getMean();
	  float empty_rms = me->getRMS();
	  if((empty_mean != 0 && empty_rms != 0) || (calib_type_ == 0)){
	    messageCounter++;
	    if(messageCounter<=message_limit_) {
	      //edm::LogError("SiPixelQualityTester::checkTestResults") << 
	      //  " *** Error for " << me->getName() << 
	      //  "," << (*ei)->getMessage() << "\n";
							
	      edm::LogWarning("SiPixelActionExecutor::checkQTestResults")  <<   " *** Error for " << me->getName() << ","
									   << (*ei)->getMessage() << " " << me->getMean() 
									   << " " << me->getRMS() 
									   << endl;
	    }
	  }
	}
	errors=vector<QReport*>();
      }
      me=0;
    }
    nval=int(); contents=vector<string>();
  }
  LogDebug("SiPixelActionExecutor::checkQTestResults") <<"messageCounter: "<<messageCounter<<" , message_limit: "<<message_limit_<<endl;
  //  if (messageCounter>=message_limit_)
  //    edm::LogWarning("SiPixelActionExecutor::checkQTestResults") << "WARNING: too many QTest failures! Giving up after "<<message_limit_<<" messages."<<endl;
  contentVec=vector<string>(); currDir=string(); messageCounter=int();
  //printing cout<<"...leaving SiPixelActionExecutor::checkQTestResults!"<<endl;
}

//=============================================================================================================
void SiPixelActionExecutor::createLayout(DQMStore * bei){
  if (configWriter_ == 0) {
    configWriter_ = new SiPixelConfigWriter();
    if (!configWriter_->init()) return;
  }
  string currDir = bei->pwd();   
  if (currDir.find("Layer") != string::npos) {
    string name = "Default";
    configWriter_->createLayout(name);
    configWriter_->createRow();
    fillLayout(bei);
  } else {
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      bei->cd(*it);
      createLayout(bei);
      bei->goUp();
    }
  }  
}

//=============================================================================================================
void SiPixelActionExecutor::fillLayout(DQMStore * bei){
	
  static int icount = 0;
  string currDir = bei->pwd();
  if (currDir.find("Ladder_") != string::npos) {
		
    vector<string> contents = bei->getMEs(); 
		
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
      if ((*im).find("Clusters") != string::npos) {
	icount++;
	if (icount != 0 && icount%6 == 0) {
	  configWriter_->createRow();
	}
	ostringstream full_path;
	full_path << "test/" << currDir << "/" << *im ;
	string element = "monitorable";
	string element_name = full_path.str();     
	configWriter_->createColumn(element, element_name);
      }
    }
  } else {
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      bei->cd(*it);
      fillLayout(bei);
      bei->goUp();
    }
  }
}

//=============================================================================================================
//
// -- Get TkMap ME names
//
int SiPixelActionExecutor::getTkMapMENames(std::vector<std::string>& names) {
  if (tkMapMENames.size() == 0) return 0;
  for (vector<string>::iterator it = tkMapMENames.begin();
       it != tkMapMENames.end(); it++) {
    names.push_back(*it) ;
  }
  return names.size();
}

//=============================================================================================================
///// Dump Module paths and IDs on screen:
void SiPixelActionExecutor::dumpModIds(DQMStore * bei, edm::EventSetup const& eSetup){
  //printing cout<<"Going to dump module IDs now!"<<endl;
  bei->cd();
  dumpBarrelModIds(bei,eSetup);
  bei->cd();
  dumpEndcapModIds(bei,eSetup);
  bei->cd();
  //printing cout<<"Done dumping module IDs!"<<endl;
}


//=============================================================================================================
void SiPixelActionExecutor::dumpBarrelModIds(DQMStore * bei, edm::EventSetup const& eSetup){
  string currDir = bei->pwd();
  string dir_name = "Ladder_";
  eSetup.get<SiPixelFedCablingMapRcd>().get(theCablingMap);
  int fedId=-1; int linkId=-1;
  if (currDir.find(dir_name) != string::npos)  {
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      if ( (*it).find("Module_") == string::npos) continue;
      bei->cd(*it);
      ndet_++;
      // long version:
      //cout<<"Ndet: "<<ndet_<<"  ,  Module: "<<bei->pwd();  
      // short version:
      cout<<bei->pwd();  
      vector<string> contents = bei->getMEs(); 
      bool first_me = false;
      int detId = -999;
      for (vector<string>::const_iterator im = contents.begin();
	   im != contents.end(); im++) {
	if(first_me) break;
	string mEName = (*im);
	string detIdString = mEName.substr((mEName.find_last_of("_"))+1,9);
	std::istringstream isst;
	isst.str(detIdString);
	if(mEName.find("_3")!=string::npos) isst>>detId;
      }
      bei->goUp();
      // long version:
      //cout<<"  , detector ID: "<<detId;
      // short version:
      cout<<" "<<detId;
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
      // long version:
      //cout<<"  , FED ID: "<<fedId<<"  , Link ID: "<<linkId<<endl;
      // short version:
      cout<<" "<<fedId<<" "<<linkId<<endl;
    }
  } else {  
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      if((*it).find("Endcap")!=string::npos) continue;
      bei->cd(*it);
      dumpBarrelModIds(bei,eSetup);
      bei->goUp();
    }
  }
}

//=============================================================================================================
void SiPixelActionExecutor::dumpEndcapModIds(DQMStore * bei, edm::EventSetup const& eSetup){
  string currDir = bei->pwd();
  string dir_name = "Panel_";
  eSetup.get<SiPixelFedCablingMapRcd>().get(theCablingMap);
  int fedId=-1; int linkId=-1;
  if (currDir.find(dir_name) != string::npos)  {
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      if ( (*it).find("Module_") == string::npos) continue;
      bei->cd(*it);
      ndet_++;
      // long version:
      //cout<<"Ndet: "<<ndet_<<"  ,  Module: "<<bei->pwd();  
      // short version:
      cout<<bei->pwd();  
      vector<string> contents = bei->getMEs(); 
      bool first_me = false;
      int detId = -999;
      for (vector<string>::const_iterator im = contents.begin();
	   im != contents.end(); im++) {
	if(first_me) break;
	string mEName = (*im);
	string detIdString = mEName.substr((mEName.find_last_of("_"))+1,9);
	std::istringstream isst;
	isst.str(detIdString);
	if(mEName.find("_3")!=string::npos) isst>>detId;
      }
      bei->goUp();
      // long version:
      //cout<<"  , detector ID: "<<detId;
      // short version:
      cout<<" "<<detId;
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
      // long version:
      //cout<<"  , FED ID: "<<fedId<<"  , Link ID: "<<linkId<<endl;
      // short version:
      cout<<" "<<fedId<<" "<<linkId<<endl;
    }
  } else {  
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      if((bei->pwd()).find("Barrel")!=string::npos) bei->goUp();
      bei->cd((*it));
      if((*it).find("Barrel")!=string::npos) continue;
      dumpEndcapModIds(bei,eSetup);
      bei->goUp();
    }
  }
	
}

//=============================================================================================================
