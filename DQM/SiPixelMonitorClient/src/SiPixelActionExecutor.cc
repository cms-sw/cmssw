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
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
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
void SiPixelActionExecutor::createSummary(DQMStore* bei, bool isUpgrade) {
  //cout<<"entering SiPixelActionExecutor::createSummary..."<<endl;
  string barrel_structure_name;
  vector<string> barrel_me_names;
  string localPath;
  if(offlineXMLfile_) localPath = string("DQM/SiPixelMonitorClient/test/sipixel_tier0_config.xml");
  else localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
//  cout<<"*********************ATTENTION! LOCALPATH= "<<localPath<<endl;
  if (configParser_ == 0) {
    configParser_ = new SiPixelConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
  if (!configParser_->getMENamesForBarrelSummary(barrel_structure_name, barrel_me_names)){
    cout << "SiPixelActionExecutor::createSummary: Failed to read Barrel Summary configuration parameters!! ";
    return;
  }
  configParser_->getSourceType(source_type_); 
//  cout<<"++++++++++++++++++++++++++SOURCE TYPE= "<<source_type_<<endl;
  bei->setCurrentFolder("Pixel/");
  //bei->cd();
  fillSummary(bei, barrel_structure_name, barrel_me_names, true, isUpgrade); // Barrel
  bei->setCurrentFolder("Pixel/");
  //bei->cd();
  string endcap_structure_name;
  vector<string> endcap_me_names;
  if (!configParser_->getMENamesForEndcapSummary(endcap_structure_name, endcap_me_names)){
    edm::LogInfo("SiPixelActionExecutor")  << "Failed to read Endcap Summary configuration parameters!! " << "\n" ;
    return;
  }

//  cout << "--- Processing endcap" << endl;

  bei->setCurrentFolder("Pixel/");
  //bei->cd();
  fillSummary(bei, endcap_structure_name, endcap_me_names, false, isUpgrade); // Endcap
  bei->setCurrentFolder("Pixel/");
  //if(!Tier0Flag_) fillDeviations(bei);
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
//  cout<<"leaving SiPixelActionExecutor::createSummary..."<<endl;
}

//=============================================================================================================
void SiPixelActionExecutor::bookDeviations(DQMStore* bei, bool isUpgrade) {
  int nBPixModules;
  if (isUpgrade) {nBPixModules=1184;} else {nBPixModules=768;} 
  
  bei->cd();
  bei->setCurrentFolder("Pixel/Barrel");
  DEV_adc_Barrel = bei->book1D("DEV_adc_Barrel","Deviation from reference;Module;<adc_ref>-<adc>",nBPixModules,0.,nBPixModules);
  DEV_ndigis_Barrel = bei->book1D("DEV_ndigis_Barrel","Deviation from reference;Module;<ndigis_ref>-<ndigis>",nBPixModules,0.,nBPixModules);
  DEV_charge_Barrel = bei->book1D("DEV_charge_Barrel","Deviation from reference;Module;<charge_ref>-<charge>",nBPixModules,0.,nBPixModules);
  DEV_nclusters_Barrel = bei->book1D("DEV_nclusters_Barrel","Deviation from reference;Module;<nclusters_ref>-<nclusters>",nBPixModules,0.,nBPixModules);
  DEV_size_Barrel = bei->book1D("DEV_size_Barrel","Deviation from reference;Module;<size_ref>-<size>",nBPixModules,0.,nBPixModules);
  bei->cd();
  bei->setCurrentFolder("Pixel/Endcap");
  DEV_adc_Endcap = bei->book1D("DEV_adc_Endcap","Deviation from reference;Module;<adc_ref>-<adc>",672,0.,672.);
  DEV_ndigis_Endcap = bei->book1D("DEV_ndigis_Endcap","Deviation from reference;Module;<ndigis_ref>-<ndigis>",672,0.,672.);
  DEV_charge_Endcap = bei->book1D("DEV_charge_Endcap","Deviation from reference;Module;<charge_ref>-<charge>",672,0.,672.);
  DEV_nclusters_Endcap = bei->book1D("DEV_nclusters_Endcap","Deviation from reference;Module;<nclusters_ref>-<nclusters>",672,0.,672.);
  DEV_size_Endcap = bei->book1D("DEV_size_Endcap","Deviation from reference;Module;<size_ref>-<size>",672,0.,672.);  
  bei->cd();
}


void SiPixelActionExecutor::fillDeviations(DQMStore* bei) {
  int n = 768;
  MonitorElement* me1; MonitorElement* me2; 
  MonitorElement* me3; MonitorElement* me4; 
  MonitorElement* me5; 
  TH1* ref1; TH1* ref2; 
  TH1* ref3; TH1* ref4; 
  TH1* ref5; 
  MonitorElement* dev1; MonitorElement* dev2; 
  MonitorElement* dev3; MonitorElement* dev4; 
  MonitorElement* dev5; 
  me1 = bei->get("Pixel/Barrel/SUMDIG_adc_Barrel");
  ref1 = me1->getRefTH1();
  dev1 = bei->get("Pixel/Barrel/DEV_adc_Barrel");
  me2 = bei->get("Pixel/Barrel/SUMDIG_ndigis_Barrel");
  ref2 = me2->getRefTH1();
  dev2 = bei->get("Pixel/Barrel/DEV_ndigis_Barrel");
  me3 = bei->get("Pixel/Barrel/SUMCLU_charge_Barrel");
  ref3 = me3->getRefTH1();
  dev3 = bei->get("Pixel/Barrel/DEV_charge_Barrel");
  me4 = bei->get("Pixel/Barrel/SUMCLU_nclusters_Barrel");
  ref4 = me4->getRefTH1();
  dev4 = bei->get("Pixel/Barrel/DEV_nclusters_Barrel");
  me5 = bei->get("Pixel/Barrel/SUMCLU_size_Barrel");
  ref5 = me5->getRefTH1();
  dev5 = bei->get("Pixel/Barrel/DEV_size_Barrel");
  for(int i=1; i!=n+1; i++){
    float ref_value; float new_value;
    // Barrel adc: 
    if(me1)if(ref1)if(dev1){
      new_value = me1->getBinContent(i);
      ref_value = ref1->GetBinContent(i); 
      dev1->setBinContent(i,ref_value-new_value);
    }
    //Barrel ndigis:
    if(me2)if(ref2)if(dev2){
      new_value = me2->getBinContent(i);
      ref_value = ref2->GetBinContent(i); 
      dev2->setBinContent(i,ref_value-new_value);
    }
    // Barrel cluster charge:
    if(me3)if(ref3)if(dev3){
      new_value = me3->getBinContent(i);
      ref_value = ref3->GetBinContent(i); 
      dev3->setBinContent(i,ref_value-new_value);
    }
    // Barrel nclusters:
    if(me4)if(ref4)if(dev4){
      new_value = me4->getBinContent(i);
      ref_value = ref4->GetBinContent(i); 
      dev4->setBinContent(i,ref_value-new_value);
    }
    // Barrel cluster size:
    if(me5)if(ref5)if(dev5){
      new_value = me5->getBinContent(i);
      ref_value = ref5->GetBinContent(i); 
      dev5->setBinContent(i,ref_value-new_value);
    }
  }

  int nn = 672;
  MonitorElement* me11; MonitorElement* me12; 
  MonitorElement* me13; MonitorElement* me14; 
  MonitorElement* me15; 
  TH1* ref11; TH1* ref12; 
  TH1* ref13; TH1* ref14; 
  TH1* ref15; 
  MonitorElement* dev11; MonitorElement* dev12; 
  MonitorElement* dev13; MonitorElement* dev14; 
  MonitorElement* dev15; 
  me11 = bei->get("Pixel/Endcap/SUMDIG_adc_Endcap");
  ref11 = me11->getRefTH1();
  dev11 = bei->get("Pixel/Endcap/DEV_adc_Endcap");
  me12 = bei->get("Pixel/Endcap/SUMDIG_ndigis_Endcap");
  ref12 = me12->getRefTH1();
  dev12 = bei->get("Pixel/Endcap/DEV_ndigis_Endcap");
  me13 = bei->get("Pixel/Endcap/SUMCLU_charge_Endcap");
  ref13 = me13->getRefTH1();
  dev13 = bei->get("Pixel/Endcap/DEV_charge_Endcap");
  me14 = bei->get("Pixel/Endcap/SUMCLU_nclusters_Endcap");
  ref14 = me14->getRefTH1();
  dev14 = bei->get("Pixel/Endcap/DEV_nclusters_Endcap");
  me15 = bei->get("Pixel/Endcap/SUMCLU_size_Endcap");
  ref15 = me15->getRefTH1();
  dev15 = bei->get("Pixel/Endcap/DEV_size_Endcap");
  for(int i=1; i!=nn+1; i++){
    float ref_value; float new_value;
    // Endcap adc: 
    if(me11)if(ref11)if(dev11){
      new_value = me11->getBinContent(i);
      ref_value = ref11->GetBinContent(i); 
      dev11->setBinContent(i,ref_value-new_value);
    }
    //Endcap ndigis:
    if(me12)if(ref12)if(dev12){
      new_value = me12->getBinContent(i);
      ref_value = ref12->GetBinContent(i); 
      dev12->setBinContent(i,ref_value-new_value);
    }
    // Endcap cluster charge:
    if(me13)if(ref13)if(dev13){
      new_value = me13->getBinContent(i);
      ref_value = ref13->GetBinContent(i); 
      dev13->setBinContent(i,ref_value-new_value);
    }
    // Endcap nclusters:
    if(me14)if(ref14)if(dev14){
      new_value = me14->getBinContent(i);
      ref_value = ref14->GetBinContent(i); 
      dev14->setBinContent(i,ref_value-new_value);
    }
    // Endcap cluster size:
    if(me15)if(ref15)if(dev15){
      new_value = me15->getBinContent(i);
      ref_value = ref15->GetBinContent(i); 
      dev15->setBinContent(i,ref_value-new_value);
    }
  }
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

void SiPixelActionExecutor::fillSummary(DQMStore* bei, string dir_name, vector<string>& me_names, bool isbarrel, bool isUpgrade)
{
	

  //cout<<"entering SiPixelActionExecutor::fillSummary..."<<endl;
  string currDir = bei->pwd();
  //cout<<"currDir= "<<currDir<<endl;
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
	temp = getSummaryME(bei, tag, isUpgrade);
	sum_mes.push_back(temp);
	tag = prefix + "_" + (*iv) + "_RMS_" 
	  + currDir.substr(currDir.find(dir_name));
	temp = getSummaryME(bei, tag, isUpgrade);
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
	  temp = getSummaryME(bei, tag, isUpgrade);
	  sum_mes.push_back(temp);
	  tag = prefix + "_" + (*iv) + "_RMS_" 
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag, isUpgrade);
	  sum_mes.push_back(temp);
	}else if((*iv) == "SiPixelErrorsCalibDigis"){
	  tag = prefix + "_" + (*iv) + "_NCalibErrors_"
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag, isUpgrade);
	  sum_mes.push_back(temp);
	}else if((*iv)=="GainFitResult2d"){
	  tag = prefix + "_" + (*iv) + "_NNegativeFits_"
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag, isUpgrade);
	  sum_mes.push_back(temp);
	}else if((*iv)=="pixelAliveSummary"){
	  tag = prefix + "_" + (*iv) + "_FracOfPerfectPix_"
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag, isUpgrade);
	  sum_mes.push_back(temp);
	  tag = prefix + "_" + (*iv) + "_mean_"
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag, isUpgrade);
	  sum_mes.push_back(temp);
	}
      }else{
	tag = prefix + "_" + (*iv) + "_" + currDir.substr(currDir.find(dir_name));
	temp = getSummaryME(bei, tag, isUpgrade);
	sum_mes.push_back(temp);
	if((*iv)=="ndigis"){
	  tag = prefix + "_" + (*iv) + "FREQ_" 
	    + currDir.substr(currDir.find(dir_name));
	  temp = getSummaryME(bei, tag, isUpgrade);
	  sum_mes.push_back(temp);
	}
	if(prefix=="SUMDIG" && (*iv)=="adc"){
	  tag = "ALLMODS_" + (*iv) + "COMB_" + currDir.substr(currDir.find(dir_name));
	  temp = bei->book1D(tag.c_str(), tag.c_str(),128, 0., 256.);
	  sum_mes.push_back(temp);
	}
	if(prefix=="SUMCLU" && (*iv)=="charge"){
	  tag = "ALLMODS_" + (*iv) + "COMB_" + currDir.substr(currDir.find(dir_name));
	  temp = bei->book1D(tag.c_str(), tag.c_str(),100, 0., 200.); // To look to get the size automatically	  
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
	      if(sname.find("_charge")!=string::npos && sname.find("Track_")==string::npos && me->getName().find("Track_")!=string::npos) continue;
	      if(sname.find("_nclusters_")!=string::npos && sname.find("Track_")==string::npos && me->getName().find("Track_")!=string::npos) continue;
	      if(sname.find("_size")!=string::npos && sname.find("Track_")==string::npos && me->getName().find("Track_")!=string::npos) continue;
//cout<<"tell me the sname and me name: "<<sname<<" , "<<me->getName()<<endl;
	      // fill summary histos:
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
	      }else if (sname.find("ALLMODS_adcCOMB_")!=string::npos || 
	                (sname.find("ALLMODS_chargeCOMB_")!=string::npos && me->getName().find("Track_")==string::npos)){
		(*isum)->getTH1F()->Add(me->getTH1F());
	      }else if (sname.find("_NErrors_")!=string::npos){
	        string path1 = fullpathname;
		path1 = path1.replace(path1.find("NErrors"),7,"errorType");
		MonitorElement * me1 = bei->get(path1);
		bool notReset=true;
	        if(me1){
	          for(int jj=1; jj<16; jj++){
	            if(me1->getBinContent(jj)>0.){
	              if(jj==6){ //errorType=30 (reset)
	                string path2 = path1;
			path2 = path2.replace(path2.find("errorType"),9,"TBMMessage");
	                MonitorElement * me2 = bei->get(path2);
	                if(me2) if(me2->getBinContent(6)>0. || me2->getBinContent(7)>0.) notReset=false;
		      }
		    }
		  }
		}
		if(notReset) (*isum)->Fill(ndet, me1->getEntries());
	      }else if ((sname.find("_charge_")!=string::npos && sname.find("Track_")==string::npos && 
	                me->getName().find("Track_")==string::npos) ||
			(sname.find("_charge_")!=string::npos && sname.find("_OnTrack_")!=string::npos && 
	                me->getName().find("_OnTrack_")!=string::npos) ||
			(sname.find("_charge_")!=string::npos && sname.find("_OffTrack_")!=string::npos && 
	                me->getName().find("_OffTrack_")!=string::npos) ||
			(sname.find("_nclusters_")!=string::npos && sname.find("Track_")==string::npos && 
	                me->getName().find("Track_")==string::npos) ||
			(sname.find("_nclusters_")!=string::npos && sname.find("_OnTrack_")!=string::npos && 
	                me->getName().find("_OnTrack_")!=string::npos) ||
			(sname.find("_nclusters_")!=string::npos && sname.find("_OffTrack_")!=string::npos && 
	                me->getName().find("_OffTrack_")!=string::npos) ||
			(sname.find("_size")!=string::npos && sname.find("Track_")==string::npos && 
	                me->getName().find("Track_")==string::npos) ||
			(sname.find("_size")!=string::npos && sname.find("_OnTrack_")!=string::npos && 
	                me->getName().find("_OnTrack_")!=string::npos) ||
			(sname.find("_size")!=string::npos && sname.find("_OffTrack_")!=string::npos && 
	                me->getName().find("_OffTrack_")!=string::npos)){
	        (*isum)->Fill(ndet, me->getMean());
	      }else if(sname.find("_charge_")==string::npos && sname.find("_nclusters_")==string::npos && sname.find("_size")==string::npos){
		(*isum)->Fill(ndet, me->getMean());
		//std::cout<<bei->pwd()<<"/"<<(*isum)->getName()<<" , "<<ndet<<" , "<<me->getMean()<<" , "<<(*isum)->getBinContent(ndet)<<std::endl;
	      }
	      
	      // set titles:
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
	      }else if (sname.find("_NErrors_")!=string::npos){
	        if(prefix=="SUMOFF" && isbarrel) title = "Total number of errors per Ladder";
		else if(prefix=="SUMOFF" && !isbarrel) title = "Total number of errors per Blade";
		else title = "Total number of errors per Module";
	      }else{
		if(prefix=="SUMOFF") title = "Mean " + sname.substr(7,(sname.find("_",7)-7)) + (isbarrel?" per Ladder":" per Blade"); 
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
	  fillSummary(bei, dir_name, me_names, true, isUpgrade); // Barrel
	  bei->goUp();
	}
	string grandbarrel_structure_name;
	vector<string> grandbarrel_me_names;
	if (!configParser_->getMENamesForGrandBarrelSummary(grandbarrel_structure_name, grandbarrel_me_names)){
	  cout << "SiPixelActionExecutor::createSummary: Failed to read Grand Barrel Summary configuration parameters!! ";
	  return;
	}
	fillGrandBarrelSummaryHistos(bei, grandbarrel_me_names, isUpgrade);
			
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
	  fillSummary(bei, dir_name, me_names, false, isUpgrade); // Endcap
	  bei->goUp();
	}
	string grandendcap_structure_name;
	vector<string> grandendcap_me_names;
	if (!configParser_->getMENamesForGrandEndcapSummary(grandendcap_structure_name, grandendcap_me_names)){
	  cout << "SiPixelActionExecutor::createSummary: Failed to read Grand Endcap Summary configuration parameters!! ";
	  return;
	}
	fillGrandEndcapSummaryHistos(bei, grandendcap_me_names, isUpgrade);
			
			
      }
  }
//  cout<<"...leaving SiPixelActionExecutor::fillSummary!"<<endl;
	
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
      bool isBooked = false;
      vector<string> contents = bei->getMEs();
      for (vector<string>::const_iterator im = contents.begin(); im != contents.end(); im++)
        if ((*im).find(*iv) != string::npos) isBooked = true;
      if(source_type_==5||source_type_==6){
	if((*iv)=="errorType"||(*iv)=="NErrors"||(*iv)=="fullType"||(*iv)=="chanNmbr"||
	   (*iv)=="TBMType"||(*iv)=="EvtNbr"||(*iv)=="evtSize"||(*iv)=="linkId"||
	   (*iv)=="ROCId"||(*iv)=="DCOLId"||(*iv)=="PXId"||(*iv)=="ROCNmbr"||
	   (*iv)=="TBMMessage"||(*iv)=="Type36Hitmap"||
	   (*iv)=="FedChLErrArray"||(*iv)=="FedChNErrArray"||(*iv)=="FedETypeNErrArray") 
	  prefix="SUMRAW";
      }
      if((*iv)=="errorType"||(*iv)=="NErrors"||(*iv)=="fullType"||(*iv)=="chanNmbr"||
	 (*iv)=="TBMType"||(*iv)=="EvtNbr"||(*iv)=="evtSize"||(*iv)=="linkId"||
	 (*iv)=="ROCId"||(*iv)=="DCOLId"||(*iv)=="PXId"||(*iv)=="ROCNmbr"||
	 (*iv)=="TBMMessage"||(*iv)=="Type36Hitmap"){
        string tag = prefix + "_" + (*iv) + "_FEDErrors";
        MonitorElement* temp = getFEDSummaryME(bei, tag);
        sum_mes.push_back(temp);
      }else if((*iv)=="FedChLErrArray"||(*iv)=="FedChNErrArray"||(*iv)=="FedETypeNErrArray"){
        string tag = prefix + "_" + (*iv);
	MonitorElement* temp;
	if((*iv)=="FedChLErrArray") {if (!isBooked) temp = bei->book2D("FedChLErrArray","Type of last error",40,-0.5,39.5,37,0.,37.);
	  else{ 
	    string fullpathname = bei->pwd() + "/" + (*iv);
	    temp = bei->get(fullpathname);
	    temp->Reset();}}  //If I don't reset this one, then I instead start adding error codes..
	if((*iv)=="FedChNErrArray") {if (!isBooked) temp = bei->book2D("FedChNErrArray","Total number of errors",40,-0.5,39.5,37,0.,37.);
	  else{ 
	    string fullpathname = bei->pwd() + "/" + (*iv);
	    temp = bei->get(fullpathname);
	    temp->Reset();}}  //If I don't reset this one, then I instead start adding error codes..
	if((*iv)=="FedETypeNErrArray"){
	  if(!isBooked){
	    temp = bei->book2D("FedETypeNErrArray","Number of each error type",40,-0.5,39.5,21,0.,21.);
	    temp->setBinLabel(1,"ROC of 25",2);
	    temp->setBinLabel(2,"Gap word",2);
	    temp->setBinLabel(3,"Dummy word",2);
	    temp->setBinLabel(4,"FIFO full",2);
	    temp->setBinLabel(5,"Timeout",2);
	    temp->setBinLabel(6,"Stack full",2);
	    temp->setBinLabel(7,"Pre-cal issued",2);
	    temp->setBinLabel(8,"Trigger clear or sync",2);
	    temp->setBinLabel(9,"No token bit",2);
	    temp->setBinLabel(10,"Overflow",2);
	    temp->setBinLabel(11,"FSM error",2);
	    temp->setBinLabel(12,"Invalid #ROCs",2);
	    temp->setBinLabel(13,"Event number",2);
	    temp->setBinLabel(14,"Slink header",2);
	    temp->setBinLabel(15,"Slink trailer",2);
	    temp->setBinLabel(16,"Event size",2);
	    temp->setBinLabel(17,"Invalid channel#",2);
	    temp->setBinLabel(18,"ROC value",2);
	    temp->setBinLabel(19,"Dcol or pixel value",2);
	    temp->setBinLabel(20,"Readout order",2);
	    temp->setBinLabel(21,"CRC error",2);
	  }
	  else{
	    string fullpathname = bei->pwd() + "/" + (*iv);
	    temp = bei->get(fullpathname);
	    temp->Reset();}  //If I don't reset this one, then I instead start adding error codes..
	}
	sum_mes.push_back(temp);
      }
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
      string fedid = (*it).substr((*it).find("_")+1);
      std::istringstream isst;
      isst.str(fedid);
      isst>>ndet; ndet++;
      vector<string> contents = bei->getMEs(); 
			
      for (vector<MonitorElement*>::const_iterator isum = sum_mes.begin();
	   isum != sum_mes.end(); isum++) {
	for (vector<string>::const_iterator im = contents.begin();
	     im != contents.end(); im++) {
	  if(((*im).find("FedChNErrArray_")!=std::string::npos && (*isum)->getName().find("FedChNErrArray")!=std::string::npos) ||
	     ((*im).find("FedChLErrArray_")!=std::string::npos && (*isum)->getName().find("FedChLErrArray")!=std::string::npos) ||
	     ((*im).find("FedETypeNErrArray_")!=std::string::npos && (*isum)->getName().find("FedETypeNErrArray")!=std::string::npos)){
	    string fullpathname = bei->pwd() + "/" + (*im); 
	    MonitorElement *  me = bei->get(fullpathname);
	    if(me && me->getIntValue()>0){
	      for(int i=0; i!=37; i++){
	        int n = (*im).find("_"); n++;
	        string channel_str = (*im).substr(n);
		std::istringstream jsst;
		jsst.str(channel_str);
		int channel=-1;
		jsst>>channel;
	        if(channel==i){
		  if((*im).find("FedETypeNErrArray_")!=std::string::npos && i<21) (*isum)->Fill(ndet-1,i,me->getIntValue());
		  else (*isum)->Fill(ndet-1,i,me->getIntValue());
		}
	      }
	    }
	  }
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
		  bool notReset=true;
	          if(me1){
	            for(int jj=1; jj<16; jj++){
	              if(me1->getBinContent(jj)>0.){
	                if(jj==6){ //errorType=30 (reset)
	                  string path2 = path1;
			  path2 = path2.replace(path2.find("errorType"),9,"TBMMessage");
	                  MonitorElement * me2 = bei->get(path2);
	                  if(me2) if(me2->getBinContent(6)>0. || me2->getBinContent(7)>0.) notReset=false; 
		        }
		      }
		    }
		  }
		  if(notReset) (*isum)->setBinContent(ndet, (*isum)->getBinContent(ndet) + me1->getEntries());
	        }else (*isum)->setBinContent(ndet, (*isum)->getBinContent(ndet) + me->getEntries());
	      }
	      (*isum)->setAxisTitle("FED #",1);
	      string title = " ";
	      title = sname.substr(7,(sname.find("_",7)-7)) + " per FED"; 
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
                                                         vector<string>& me_names,
                                                         bool isUpgrade) {
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
	  if (first_subdir && !isUpgrade){
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
	  } else if (first_subdir && isUpgrade){
//	    bool create_me = true;
	    nbin = me->getTH1F()->GetNbinsX();        
	    string me_name = prefix + "_" + (*iv) + "_" + dir_name;
	    if((*iv)=="adcCOMB"||(*iv)=="chargeCOMB") me_name = "ALLMODS_" + (*iv) + "_" + dir_name;
	    else if(prefix=="SUMOFF" && dir_name=="Barrel") nbin=296;
	    else if((*iv)=="adcCOMB") nbin=256;
	    else if(dir_name=="Barrel") nbin=1184;
	    else if(prefix=="SUMOFF" && dir_name.find("Shell")!=string::npos) nbin=74;
	    else if(dir_name.find("Shell")!=string::npos) nbin=296;
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
	      if((*igm)->getName().find("NErrors_") != string::npos && prefix=="SUMOFF") title = "Total number of errors per Ladder";
	      else if((*igm)->getName().find("NErrors_") != string::npos && prefix=="SUMRAW") title = "Total number of errors per Module";
	      else if(prefix=="SUMOFF") title = "mean " + (*iv) + " per Ladder"; 
	      else if((*igm)->getName().find("FREQ_") != string::npos && prefix!="SUMOFF") title = "NEvents with digis per Module"; 
	      else if((*igm)->getName().find("FREQ_") != string::npos && prefix=="SUMOFF") title = "NEvents with digis per Ladder/Blade"; 
	      else if((*igm)->getName().find("adcCOMB_") != string::npos) title = "NDigis";
	      else if((*igm)->getName().find("chargeCOMB_") != string::npos) title = "NClusters";
	      else title = "mean " + (*iv) + " per Module"; 
	      (*igm)->setAxisTitle(title,2);
		  
		  // Setting binning
	      if (!isUpgrade) {
	      if((*igm)->getName().find("ALLMODS_adcCOMB_")!=string::npos){
		nbin_subdir=128;
	      }else if((*igm)->getName().find("ALLMODS_chargeCOMB_")!=string::npos){
		nbin_subdir=100;
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
	      } else if (isUpgrade) {
	        if((*igm)->getName().find("ALLMODS_adcCOMB_")!=string::npos){
		  nbin_subdir=128;
	        }else if((*igm)->getName().find("ALLMODS_chargeCOMB_")!=string::npos){
		  nbin_subdir=100;
	        }else if((*igm)->getName().find("Ladder") != string::npos){
		  nbin_i=0; nbin_subdir=4;
	        }else if((*igm)->getName().find("Layer") != string::npos){
		  nbin_i=(cnt-1)*4; nbin_subdir=4;
	        }else if((*igm)->getName().find("Shell") != string::npos){
		  if(prefix!="SUMOFF"){
		   if(iDir==0){ nbin_i=0; nbin_subdir=24; }//40(2*20)-->24(2*12)
		    else if(iDir==1){ nbin_i=24; nbin_subdir=56; }//64(32*2)-->56(2*28)
		    else if(iDir==2){ nbin_i=80; nbin_subdir=88; }//88(44*2)-->same88(44*2)
		    else if(iDir==3){ nbin_i=168; nbin_subdir=128; }
		  }else{
		    if(iDir==0){ nbin_i=0; nbin_subdir=6; }//10-->6
		    else if(iDir==1){ nbin_i=6; nbin_subdir=14; }//16-->14
		    else if(iDir==2){ nbin_i=20; nbin_subdir=22; }//22-->same22
		    else if(iDir==3){ nbin_i=42; nbin_subdir=32; }
		  }
	        }else if((*igm)->getName().find("Barrel") != string::npos){
		  if(prefix!="SUMOFF"){
		    if(iDir==0){ nbin_i=0; nbin_subdir=296; }//192=76 8/4-->296=1184/4
		    else if(iDir==1){ nbin_i=296; nbin_subdir=296; }//296*2,*3,*4=1184
		    else if(iDir==2){ nbin_i=592; nbin_subdir=296; }
		    else if(iDir==3){ nbin_i=888; nbin_subdir=296; }
		    else if(iDir==4){ nbin_i=1184; nbin_subdir=296; }
		  }else{
		    if(iDir==0){ nbin_i=0; nbin_subdir=74; }//48=192/4-->74=296/4
		    else if(iDir==1){ nbin_i=74; nbin_subdir=74; }//74*2,...*4=296
		    else if(iDir==2){ nbin_i=148; nbin_subdir=74; }
		    else if(iDir==3){ nbin_i=222; nbin_subdir=74; }
		    else if(iDir==4){ nbin_i=296; nbin_subdir=74; }
		  }
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
                                                         vector<string>& me_names,
                                                         bool isUpgrade) {
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
	  if (first_subdir && !isUpgrade){
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
	  } else if(first_subdir && isUpgrade){
            nbin = me->getTH1F()->GetNbinsX();        
	    string me_name = prefix + "_" + (*iv) + "_" + dir_name;
	    if((*iv)=="adcCOMB"||(*iv)=="chargeCOMB") me_name = "ALLMODS_" + (*iv) + "_" + dir_name;
	    else if(prefix=="SUMOFF" && dir_name=="Endcap") nbin=336;
	    else if(dir_name=="Endcap") nbin=672;
	    else if(prefix=="SUMOFF" && dir_name.find("HalfCylinder")!=string::npos) nbin=84;
	    else if(dir_name.find("HalfCylinder")!=string::npos) nbin=168;
	    else if(prefix=="SUMOFF" && dir_name.find("Disk")!=string::npos) nbin=28;
	    else if(dir_name.find("Disk")!=string::npos) nbin=56;
	    else if(dir_name.find("Blade")!=string::npos) nbin=2;
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
	      if((*igm)->getName().find("NErrors_") != string::npos && prefix=="SUMOFF") title = "Total number of errors per Blade";
	      else if((*igm)->getName().find("NErrors_") != string::npos && prefix=="SUMRAW") title = "Total number of errors per Module";
	      else if(prefix=="SUMOFF") title = "mean " + (*iv) + " per Blade"; 
	      else if((*igm)->getName().find("FREQ_") != string::npos) title = "NEvents with digis per Module"; 
	      else if((*igm)->getName().find("adcCOMB_")!=string::npos) title = "NDigis";
	      else if((*igm)->getName().find("chargeCOMB_")!=string::npos) title = "NClusters";
	      else title = "mean " + (*iv) + " per Module"; 
	      (*igm)->setAxisTitle(title,2);
	      nbin_i=0; 
	      if (!isUpgrade) {
	      if((*igm)->getName().find("ALLMODS_adcCOMB_")!=string::npos){
		nbin_subdir=128;
	      }else if((*igm)->getName().find("ALLMODS_chargeCOMB_")!=string::npos){
		nbin_subdir=100;
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
              } else if (isUpgrade) {
                if((*igm)->getName().find("ALLMODS_adcCOMB_")!=string::npos){
		  nbin_subdir=128;
	        }else if((*igm)->getName().find("ALLMODS_chargeCOMB_")!=string::npos){
		  nbin_subdir=100;
	        }else if((*igm)->getName().find("Panel_") != string::npos){
		  nbin_subdir=2;
		  //	        }else if((*igm)->getName().find("Panel_1") != string::npos){
		  //		  nbin_subdir=4;
		  //	        }else if((*igm)->getName().find("Panel_2") != string::npos){
		  //		  nbin_subdir=3;
	        }else if((*igm)->getName().find("Blade") != string::npos){
		  if((*im).find("_1") != string::npos) nbin_subdir=1;
		  if((*im).find("_2") != string::npos) {nbin_i=1; nbin_subdir=1;}
	        }else if((*igm)->getName().find("Disk") != string::npos){
		  nbin_i=((cnt-1)%28)*2; nbin_subdir=2;
	        }else if((*igm)->getName().find("HalfCylinder") != string::npos){
		  if(prefix!="SUMOFF"){
		    nbin_subdir=56;
		    if((*im).find("_2") != string::npos) nbin_i=56;
                    if((*im).find("_3") != string::npos) nbin_i=112;
		  }else{
		    nbin_subdir=28;
		    if((*im).find("_2") != string::npos) nbin_i=28;
                    if((*im).find("_3") != string::npos) nbin_i=56;
		  }
	        }else if((*igm)->getName().find("Endcap") != string::npos){
		  if(prefix!="SUMOFF"){
		    nbin_subdir=168;
		    if((*im).find("_mO") != string::npos) nbin_i=168;
		    if((*im).find("_pI") != string::npos) nbin_i=336;
		    if((*im).find("_pO") != string::npos) nbin_i=504;
		  }else{
		    nbin_subdir=84;
		    if((*im).find("_mO") != string::npos) nbin_i=84;
		    if((*im).find("_pI") != string::npos) nbin_i=168;
		    if((*im).find("_pO") != string::npos) nbin_i=252;
		  }
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
  if(me_name.find("ALLMODS_adcCOMB_")!=string::npos) temp_me = bei->book1D(me_name.c_str(),me_name.c_str(),128,0,256);
  else if(me_name.find("ALLMODS_chargeCOMB_")!=string::npos) temp_me = bei->book1D(me_name.c_str(),me_name.c_str(),100,0,200);
  else temp_me = bei->book1D(me_name.c_str(),me_name.c_str(),nbin,1.,nbin+1.);
  if (temp_me) mes.push_back(temp_me);
	
  //  if(temp_me) cout<<"finally found grand ME: "<<me_name<<endl;
}


//=============================================================================================================
//
// -- Get Summary ME
//
MonitorElement* SiPixelActionExecutor::getSummaryME(DQMStore* bei,
                                                    string me_name,
                                                    bool isUpgrade) {
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
  if (!isUpgrade) {
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
  }//endifNOTUpgrade
  else if (isUpgrade) {
    if(me_name.find("SUMOFF")==string::npos){
      if(me_name.find("Blade_")!=string::npos)me = bei->book1D(me_name.c_str(), me_name.c_str(),2,1.,3.);
      else me = bei->book1D(me_name.c_str(), me_name.c_str(),1,1.,2.);
      //      if(me_name.find("Panel_2")!=string::npos)  me = bei->book1D(me_name.c_str(), me_name.c_str(),3,1.,4.);
      //      else me = bei->book1D(me_name.c_str(), me_name.c_str(),4,1.,5.);
    }else if(me_name.find("Layer_1")!=string::npos){ me = bei->book1D(me_name.c_str(), me_name.c_str(),6,1.,7.);
    }else if(me_name.find("Layer_2")!=string::npos){ me = bei->book1D(me_name.c_str(), me_name.c_str(),14,1.,15.);
    }else if(me_name.find("Layer_3")!=string::npos){ me = bei->book1D(me_name.c_str(), me_name.c_str(),22,1.,23.);
    }else if(me_name.find("Layer_4")!=string::npos){ me = bei->book1D(me_name.c_str(), me_name.c_str(),32,1.,33.);
    }else if(me_name.find("Disk_")!=string::npos){ me = bei->book1D(me_name.c_str(), me_name.c_str(),28,1.,29.);
    }
  }//endifUpgrade
  
	
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
      if((*it).find("Ladder_")!=string::npos) continue;
      if((*it).find("Blade_")!=string::npos) continue;
      if((*it).find("Layer_")!=string::npos) continue;
      if((*it).find("Disk_")!=string::npos) continue;
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
	
  if(dname.find("Layer_")!=string::npos || dname.find("Disk_")!=string::npos){ 
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
		for(int i=1; i!=me->getNbinsX()+1; i++) for(int j=1; j!=me->getNbinsY()+1; j++){
		  float previous = OccupancyMap->getBinContent(i,j);
		  OccupancyMap->setBinContent(i,j,previous + me->getBinContent(i,j));
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
// -- Tracker Map
//

void SiPixelActionExecutor::bookTrackerMaps(DQMStore* bei, std::string name)
{
	bei->setCurrentFolder("Pixel/Barrel");
	std::string partB[] = { "Layer_1", "Layer_2", "Layer_3"};
	bei->book2D("TRKMAP_" + name + "_" + partB[0], "TRKMAP_" + name + "_" + partB[0], 20, 1., 21., 8, 1., 9.);
	bei->book2D("TRKMAP_" + name + "_" + partB[1], "TRKMAP_" + name + "_" + partB[1], 32, 1., 33., 8, 1., 9.);
	bei->book2D("TRKMAP_" + name + "_" + partB[2], "TRKMAP_" + name + "_" + partB[2], 44, 1., 45., 8, 1., 9.);
		
	bei->setCurrentFolder("Pixel/Endcap");
	std::string partE[] = { "Disc_1_M", "Disc_2_M", "Disc_1_P", "Disc_2_P" };
	for(Int_t p = 0 ; p < NCyl ; p++)
		bei->book2D("TRKMAP_" + name + "_" + partE[p], "TRKMAP_" + name + "_" + partE[p], 24, 1., 25., 7, 1., 8.);
}


void SiPixelActionExecutor::createMaps(DQMStore* bei, std::string type, std::string name, funcType ff)
{
//	cout << "Starting with SiPixelActionExecutor::createMaps" << endl;

	Double_t mapB[NLev1][NLev2][NLev3][NLev4];
	bei->setCurrentFolder("Pixel/Barrel/");
	createMap(mapB, type, bei, ff, true);
	Double_t minB = mapMin(mapB, true);
	Double_t maxB = mapMax(mapB, true);
	
	
	Double_t mapE[NLev1][NLev2][NLev3][NLev4];
	bei->setCurrentFolder("Pixel/Endcap/");
	createMap(mapE, type, bei, ff, false);
	Double_t minE = mapMin(mapE, false);
	Double_t maxE = mapMax(mapE, false);
	
	Double_t min = minE<=minB?minE:minB;
	Double_t max = maxE>=maxB?maxE:maxB;
	if(!min)
		min = -0.01;
	
	//	prephistosB(histB, mapB, type, min, max);
	//	prephistosE(histE, mapE, type, min, max);


	MonitorElement* meB[NLayer];
	//	TH2F* histB[NLayer];
	MonitorElement* meE[NCyl];
	//	TH2F* histE[NCyl];
	bei->setCurrentFolder("Pixel/Barrel");
	prephistosB(meB, bei, mapB, name, min, max);
	bei->setCurrentFolder("Pixel/Endcap");
	prephistosE(meE, bei, mapE, name, min, max);

//	cout << "Done with SiPixelActionExecutor::createMaps" << endl;
}

//=============================================================================================================

int SiPixelActionExecutor::createMap(Double_t map[][NLev2][NLev3][NLev4], std::string type, DQMStore* bei, funcType ff, bool isBarrel)
{
	// cout << "Starting with SiPixelActionExecutor::createMap" << endl;
	//int createMap(Double_t map[][NLev2][NLev3][NLev4], TString type, TDirectoryFile* dirMain, funcType ff){
	vector<string> dirLev1 = bei->getSubdirs();
	Int_t i = 0;
	for (vector<string>::const_iterator it = dirLev1.begin(); it != dirLev1.end(); it++) // goes over HalfCylinders in Endcap and over Shells in Barrel
	{
		//cout << "Current Directory: " << *it << endl;
		bei->cd(*it);
		vector<string> dirLev2 = bei->getSubdirs();
		Int_t j = 0;
		for (vector<string>::const_iterator it2 = dirLev2.begin(); it2 != dirLev2.end(); it2++) // goes over Disks in Endcap and over Layers in Barrel
		{
			//cout << "Current Directory: " << *it2 << endl;
			bei->cd(*it2);
			Int_t k = 0;
			vector<string> dirLev3 = bei->getSubdirs();
			for (vector<string>::const_iterator it3 = dirLev3.begin(); it3 != dirLev3.end(); it3++) // goes over Blades in Endcap and over Ladders in Barrel
			{
				//cout << "Current Directory: " << *it3 << endl;
				bei->cd(*it3);
				if(Tier0Flag_)
					for (Int_t l = 0; l < NLev4; l++)
						getData(map, type, bei, ff, i, j, k, l);
				else
				{
					Int_t l = 0;
					vector<string> dirLev4 = bei->getSubdirs();
					for (vector<string>::const_iterator it4 = dirLev4.begin(); it4 != dirLev4.end(); it4++)
					{
						// cout << "Current Directory: " << *it4 << endl;
						bei->cd(*it4);
						if (isBarrel)
							getData(map, type, bei, ff, i, j, k, l++);
						else
						{
							vector<string> dirLev5 = bei->getSubdirs();
							for (vector<string>::const_iterator it5 = dirLev5.begin(); it5 != dirLev5.end(); it5++)
							{
								// cout << "Current Directory: " << *it5 << endl;
								bei->cd(*it5);
								getData(map, type, bei, ff, i, j, k, l++);
							}
						}
					}
				}
				k++;
			}
			j++;
		}
		i++;
	}
	
	// cout << "Done with SiPixelActionExecutor::createMap" << endl;
	return 0;
}

//=============================================================================================================

void SiPixelActionExecutor::getData(Double_t map[][NLev2][NLev3][NLev4], std::string type, DQMStore* bei, funcType ff, Int_t i, Int_t j, Int_t k, Int_t l) {
	
//	cout << "Starting with SiPixelActionExecutor::getData" << endl;
	vector<string> contents = bei->getMEs();
	for (vector<string>::const_iterator im = contents.begin(); im != contents.end(); im++)
	{
		if((*im).find(type + "_") == string::npos){
			// cout << "Skip";
			continue; // Searching for specific type
		}
		//cout << "Name: "  << *im << endl;
		std::string fullpathname = bei->pwd() + "/" + (*im);	
		MonitorElement*  me = bei->get(fullpathname);
		
		if (me) {
		TH1F* histo = me->getTH1F();
		
		Int_t nbins = histo->GetNbinsX();
		// cout << "# of bins: " << nbins << endl;
		switch (ff){
			case EachBinContent:
				map[i][j][k][l] = histo->GetBinContent(l + 1);
				break;
				
			case Entries:
				map[i][j][k][l] = histo->GetEntries();
				break;
					
			case Mean:
				map[i][j][k][l] = histo->GetMean();
				break;
							
			case Sum:
			{
				Double_t sum = 0;
				for(Int_t m = 0; m < nbins; m++)
					sum += histo->GetBinContent(m + 1);
				map[i][j][k][l] = sum;
			}
				break;
							
			case WeightedSum:
			{
				Double_t sum = 0;
				for(Int_t m = 0; m < nbins; m++)
					sum += histo->GetBinContent(m + 1) * histo->GetBinLowEdge(m + 1);
				map[i][j][k][l] = sum;
			}
				break;
				
			default:
					map[i][j][k][l] = 0;
		}
	}}
//	cout << "Done with SiPixelActionExecutor::getData" << endl;
}

//=============================================================================================================

void SiPixelActionExecutor::prephistosB(MonitorElement* me[NLayer], DQMStore *bei, const Double_t map[][NLev2][NLev3][NLev4],std::string name, Double_t min, Double_t max){
	// cout << "Starting with SiPixelActionExecutor::prephistosB" << endl;
	std::string part[] = { "Layer_1", "Layer_2", "Layer_3"};
	std::string path = bei->pwd();
	for (Int_t i = 0; i < NLayer; i++)
	{
		std::string fullpath = path + "/" + "TRKMAP_" + name + "_" + part[i];
		MonitorElement* temp = bei->get(fullpath);
		if(temp)
			me[i] = temp;
		else
			cout << "Problem: " << fullpath << endl;
	}
	
	for(Int_t p = 0 ; p < NLayer ; p++){
		for(Int_t b = 0 ; b < (10 + 6 * p); b++)
			for(Int_t i = 0 ; i < NModuleB ; i++){
				me[p]->getTH2F()->SetBinContent(b + 1, i + 1, map[0][p][b][i]);
				me[p]->getTH2F()->SetBinContent(b + 1, i + 1 + NModuleB, map[1][p][b][i]);
				me[p]->getTH2F()->SetBinContent(b + 1 + 10 + 6 * p, i + 1, map[2][p][b][i]);
				me[p]->getTH2F()->SetBinContent(b + 1 + 10 + 6 * p, i + 1 + NModuleB, map[3][p][b][i]);
			}
		me[p]->getTH2F()->SetMinimum(min);
		me[p]->getTH2F()->SetMaximum(max);
	}
	// cout << "Done with SiPixelActionExecutor::prephistosB" << endl;
}

//=============================================================================================================

void SiPixelActionExecutor::prephistosE(MonitorElement* me[NCyl], DQMStore *bei, const Double_t map[][NLev2][NLev3][NLev4], std::string name, Double_t min, Double_t max){
	// cout << "Starting with SiPixelActionExecutor::prephistosE" << endl;
	std::string part[] = { "Disc_1_M", "Disc_2_M", "Disc_1_P", "Disc_2_P" };
	std::string path = bei->pwd();
	for (Int_t i = 0; i < NCyl; i++)
	{
		std::string fullpath = path + "/" + "TRKMAP_" + name + "_" + part[i];
		MonitorElement* temp = bei->get(fullpath);
		if(temp)
		{
			me[i] = temp;
			me[i]->getTH2F()->SetMinimum(min);
			me[i]->getTH2F()->SetMaximum(max);
		}
		else
			cout << "Problem: " << fullpath << endl;
	}

	for(Int_t c = 0 ; c < NCyl ; c += 2)
		for(Int_t d = 0 ; d < NDisk ; d++)
			for(Int_t b = 0 ; b < NBlade ; b++){
				me[c + d]->getTH2F()->SetBinContent(b + 1, 1, map[c][d][b][0]);
				me[c + d]->getTH2F()->SetBinContent(b + 1, 2, map[c][d][b][4]);
				me[c + d]->getTH2F()->SetBinContent(b + 1, 3, map[c][d][b][1]);
				me[c + d]->getTH2F()->SetBinContent(b + 1, 4, map[c][d][b][5]);
				me[c + d]->getTH2F()->SetBinContent(b + 1, 5, map[c][d][b][2]);
				me[c + d]->getTH2F()->SetBinContent(b + 1, 6, map[c][d][b][6]);
				me[c + d]->getTH2F()->SetBinContent(b + 1, 7, map[c][d][b][3]);
				
				me[c + d]->getTH2F()->SetBinContent(2 * NBlade - b, 1, map[c + 1][d][b][0]);
				me[c + d]->getTH2F()->SetBinContent(2 * NBlade - b, 2, map[c + 1][d][b][4]);
				me[c + d]->getTH2F()->SetBinContent(2 * NBlade - b, 3, map[c + 1][d][b][1]);
				me[c + d]->getTH2F()->SetBinContent(2 * NBlade - b, 4, map[c + 1][d][b][5]);
				me[c + d]->getTH2F()->SetBinContent(2 * NBlade - b, 5, map[c + 1][d][b][2]);
				me[c + d]->getTH2F()->SetBinContent(2 * NBlade - b, 6, map[c + 1][d][b][6]);
				me[c + d]->getTH2F()->SetBinContent(2 * NBlade - b, 7, map[c + 1][d][b][3]);
			}
	// cout << "Done with SiPixelActionExecutor::prephistosE" << endl;
}


//=============================================================================================================
Double_t SiPixelActionExecutor::mapMin(const Double_t map[][NLev2][NLev3][NLev4], bool isBarrel){
	
	Double_t min = map[0][0][0][0];
	
	for(Int_t p = 0 ; p < NLev1 ; p++)
		for(Int_t d = 0 ; d < (isBarrel?3:2) ; d++)
			for(Int_t b = 0 ; b < (isBarrel?(d*6+10):12) ; b++)
				for(Int_t i = 0 ; i < (isBarrel?4:7); i++){
					if(map[p][d][b][i] < min)
						min = map[p][d][b][i];
				}
	// cout << "Done with SiPixelActionExecutor::mapMin" << endl;
	return min;
}

//=============================================================================================================

Double_t SiPixelActionExecutor::mapMax(const Double_t map[][NLev2][NLev3][NLev4], bool isBarrel){
	
	Double_t max = map[0][0][0][0];
	
	for(Int_t p = 0 ; p < NLev1 ; p++)
		for(Int_t d = 0 ; d < (isBarrel?3:2); d++)
			for(Int_t b = 0 ; b < (isBarrel?(d*6+10):12); b++)
				for(Int_t i = 0 ; i < (isBarrel?4:7); i++)
					if(map[p][d][b][i] > max)
						max = map[p][d][b][i];
	// cout << "Done with SiPixelActionExecutor::mapMax" << endl;
	return max;
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
      assert(detId >= 0);
      sipixelobjects::DetectorIndex detector = {static_cast<unsigned int>(detId), 1, 1};	   
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
      assert(detId >= 0);
      sipixelobjects::DetectorIndex detector = {static_cast<unsigned int>(detId), 1, 1};	   
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
///// Dump Module paths and IDs on screen:
void SiPixelActionExecutor::dumpRefValues(DQMStore * bei, edm::EventSetup const& eSetup){
  //printing cout<<"Going to dump module IDs now!"<<endl;
  bei->cd();
  dumpBarrelRefValues(bei,eSetup);
  bei->cd();
  dumpEndcapRefValues(bei,eSetup);
  bei->cd();
  //printing cout<<"Done dumping module IDs!"<<endl;
}


//=============================================================================================================
void SiPixelActionExecutor::dumpBarrelRefValues(DQMStore * bei, edm::EventSetup const& eSetup){
  MonitorElement* me;
  me = bei->get("Pixel/Barrel/SUMDIG_adc_Barrel");
  if(me){
    std::cout<<"SUMDIG_adc_Barrel: "<<std::endl;
    for(int i=1; i!=769; i++) std::cout<<i<<" "<<me->getBinContent(i)<<std::endl;
  }
  me = bei->get("Pixel/Barrel/SUMDIG_ndigis_Barrel");
  if(me){
    std::cout<<"SUMDIG_ndigis_Barrel: "<<std::endl;
    for(int i=1; i!=769; i++) std::cout<<i<<" "<<me->getBinContent(i)<<std::endl;
  }
  me = bei->get("Pixel/Barrel/SUMCLU_charge_Barrel");
  if(me){
    std::cout<<"SUMCLU_charge_Barrel: "<<std::endl;
    for(int i=1; i!=769; i++) std::cout<<i<<" "<<me->getBinContent(i)<<std::endl;
  }
  me = bei->get("Pixel/Barrel/SUMCLU_nclusters_Barrel");
  if(me){
    std::cout<<"SUMCLU_nclusters_Barrel: "<<std::endl;
    for(int i=1; i!=769; i++) std::cout<<i<<" "<<me->getBinContent(i)<<std::endl;
  }
  me = bei->get("Pixel/Barrel/SUMCLU_size_Barrel");
  if(me){
    std::cout<<"SUMCLU_size_Barrel: "<<std::endl;
    for(int i=1; i!=769; i++) std::cout<<i<<" "<<me->getBinContent(i)<<std::endl;
  }
}

//=============================================================================================================
void SiPixelActionExecutor::dumpEndcapRefValues(DQMStore * bei, edm::EventSetup const& eSetup){
  MonitorElement* me;
  me = bei->get("Pixel/Endcap/SUMDIG_adc_Endcap");
  if(me){
    std::cout<<"SUMDIG_adc_Endcap: "<<std::endl;
    for(int i=1; i!=673; i++) std::cout<<i<<" "<<me->getBinContent(i)<<std::endl;
  }
  me = bei->get("Pixel/Endcap/SUMDIG_ndigis_Endcap");
  if(me){
    std::cout<<"SUMDIG_ndigis_Endcap: "<<std::endl;
    for(int i=1; i!=673; i++) std::cout<<i<<" "<<me->getBinContent(i)<<std::endl;
  }
  me = bei->get("Pixel/Endcap/SUMCLU_charge_Endcap");
  if(me){
    std::cout<<"SUMCLU_charge_Endcap: "<<std::endl;
    for(int i=1; i!=673; i++) std::cout<<i<<" "<<me->getBinContent(i)<<std::endl;
  }
  me = bei->get("Pixel/Endcap/SUMCLU_nclusters_Endcap");
  if(me){
    std::cout<<"SUMCLU_nclusters_Endcap: "<<std::endl;
    for(int i=1; i!=673; i++) std::cout<<i<<" "<<me->getBinContent(i)<<std::endl;
  }
  me = bei->get("Pixel/Endcap/SUMCLU_size_Endcap");
  if(me){
    std::cout<<"SUMCLU_size_Endcap: "<<std::endl;
    for(int i=1; i!=673; i++) std::cout<<i<<" "<<me->getBinContent(i)<<std::endl;
  }
}

//=============================================================================================================

void SiPixelActionExecutor::bookEfficiency(DQMStore * bei, bool isUpgrade){
  // Barrel
  bei->cd();
  bei->setCurrentFolder("Pixel/Barrel");
  if (!isUpgrade) {
  if(Tier0Flag_){
    HitEfficiency_L1 = bei->book2D("HitEfficiency_L1","Hit Efficiency in Barrel_Layer1;Module;Ladder",8,-4,4,20,-10.,10.);
    HitEfficiency_L2 = bei->book2D("HitEfficiency_L2","Hit Efficiency in Barrel_Layer2;Module;Ladder",8,-4,4,32,-16.,16.);
    HitEfficiency_L3 = bei->book2D("HitEfficiency_L3","Hit Efficiency in Barrel_Layer3;Module;Ladder",8,-4,4,44,-22.,22.);
  }else{
    HitEfficiency_L1 = bei->book2D("HitEfficiency_L1","Hit Efficiency in Barrel_Layer1;Module;Ladder",8,-4.,4.,20,-10.,10.);
    HitEfficiency_L2 = bei->book2D("HitEfficiency_L2","Hit Efficiency in Barrel_Layer2;Module;Ladder",8,-4.,4.,32,-16.,16.);
    HitEfficiency_L3 = bei->book2D("HitEfficiency_L3","Hit Efficiency in Barrel_Layer3;Module;Ladder",8,-4.,4.,44,-22.,22.);
  }
  }//endifNOTUpgrade
  else if (isUpgrade) {
      if(Tier0Flag_){
      HitEfficiency_L1 = bei->book2D("HitEfficiency_L1","Hit Efficiency in Barrel_Layer1;z-side;Ladder",2,-1.,1.,12,-6.,6.);
      HitEfficiency_L2 = bei->book2D("HitEfficiency_L2","Hit Efficiency in Barrel_Layer2;z-side;Ladder",2,-1.,1.,28,-14.,14.);
      HitEfficiency_L3 = bei->book2D("HitEfficiency_L3","Hit Efficiency in Barrel_Layer3;z-side;Ladder",2,-1.,1.,44,-22.,22.);
      HitEfficiency_L4 = bei->book2D("HitEfficiency_L4","Hit Efficiency in Barrel_Layer4;z-side;Ladder",2,-1.,1.,64,-32.,32.);
    }else{
      HitEfficiency_L1 = bei->book2D("HitEfficiency_L1","Hit Efficiency in Barrel_Layer1;Module;Ladder",8,-4.,4.,12,-6.,6.);
      HitEfficiency_L2 = bei->book2D("HitEfficiency_L2","Hit Efficiency in Barrel_Layer2;Module;Ladder",8,-4.,4.,28,-14.,14.);
      HitEfficiency_L3 = bei->book2D("HitEfficiency_L3","Hit Efficiency in Barrel_Layer3;Module;Ladder",8,-4.,4.,44,-22.,22.);
      HitEfficiency_L4 = bei->book2D("HitEfficiency_L4","Hit Efficiency in Barrel_Layer4;Module;Ladder",8,-4.,4.,64,-32.,32.);
    }
  }//endifUpgrade
  // Endcap
  bei->cd();
  bei->setCurrentFolder("Pixel/Endcap");
  if (!isUpgrade) {
  if(Tier0Flag_){
    HitEfficiency_Dp1 = bei->book2D("HitEfficiency_Dp1","Hit Efficiency in Endcap_Disk_p1;Blades;",24,-12.,12.,1,0.,1.);
    HitEfficiency_Dp2 = bei->book2D("HitEfficiency_Dp2","Hit Efficiency in Endcap_Disk_p2;Blades;",24,-12.,12.,1,0.,1.);
    HitEfficiency_Dm1 = bei->book2D("HitEfficiency_Dm1","Hit Efficiency in Endcap_Disk_m1;Blades;",24,-12.,12.,1,0.,1.);
    HitEfficiency_Dm2 = bei->book2D("HitEfficiency_Dm2","Hit Efficiency in Endcap_Disk_m2;Blades;",24,-12.,12.,1,0.,1.);
  }else{
    HitEfficiency_Dp1 = bei->book2D("HitEfficiency_Dp1","Hit Efficiency in Endcap_Disk_p1;Blades;Modules",24,-12.,12.,7,1.,8.);
    HitEfficiency_Dp2 = bei->book2D("HitEfficiency_Dp2","Hit Efficiency in Endcap_Disk_p2;Blades;Modules",24,-12.,12.,7,1.,8.);
    HitEfficiency_Dm1 = bei->book2D("HitEfficiency_Dm1","Hit Efficiency in Endcap_Disk_m1;Blades;Modules",24,-12.,12.,7,1.,8.);
    HitEfficiency_Dm2 = bei->book2D("HitEfficiency_Dm2","Hit Efficiency in Endcap_Disk_m2;Blades;Modules",24,-12.,12.,7,1.,8.);
  }
  } else if (isUpgrade) {
    if(Tier0Flag_){
      HitEfficiency_Dp1 = bei->book2D("HitEfficiency_Dp1","Hit Efficiency in Endcap_Disk_p1;Blades;",28,-17.,11.,1,0.,1.);
      HitEfficiency_Dp2 = bei->book2D("HitEfficiency_Dp2","Hit Efficiency in Endcap_Disk_p2;Blades;",28,-17.,11.,1,0.,1.);
      HitEfficiency_Dp3 = bei->book2D("HitEfficiency_Dp3","Hit Efficiency in Endcap_Disk_p3;Blades;",28,-17.,11.,1,0.,1.);
      HitEfficiency_Dm1 = bei->book2D("HitEfficiency_Dm1","Hit Efficiency in Endcap_Disk_m1;Blades;",28,-17.,11.,1,0.,1.);
      HitEfficiency_Dm2 = bei->book2D("HitEfficiency_Dm2","Hit Efficiency in Endcap_Disk_m2;Blades;",28,-17.,11.,1,0.,1.);
      HitEfficiency_Dm3 = bei->book2D("HitEfficiency_Dm3","Hit Efficiency in Endcap_Disk_m3;Blades;",28,-17.,11.,1,0.,1.);
    }else{
      HitEfficiency_Dp1 = bei->book2D("HitEfficiency_Dp1","Hit Efficiency in Endcap_Disk_p1;Blades;Modules",28,-17.,11.,2,1.,3.);
      HitEfficiency_Dp2 = bei->book2D("HitEfficiency_Dp2","Hit Efficiency in Endcap_Disk_p2;Blades;Modules",28,-17.,11.,2,1.,3.);
      HitEfficiency_Dp3 = bei->book2D("HitEfficiency_Dp3","Hit Efficiency in Endcap_Disk_p3;Blades;Modules",28,-17.,11.,2,1.,3.);
      HitEfficiency_Dm1 = bei->book2D("HitEfficiency_Dm1","Hit Efficiency in Endcap_Disk_m1;Blades;Modules",28,-17.,11.,2,1.,3.);
      HitEfficiency_Dm2 = bei->book2D("HitEfficiency_Dm2","Hit Efficiency in Endcap_Disk_m2;Blades;Modules",28,-17.,11.,2,1.,3.);
      HitEfficiency_Dm3 = bei->book2D("HitEfficiency_Dm3","Hit Efficiency in Endcap_Disk_m3;Blades;Modules",28,-17.,11.,2,1.,3.);
    }
  }//endif(isUpgrade)
}

//=============================================================================================================

void SiPixelActionExecutor::createEfficiency(DQMStore * bei, bool isUpgrade){
  //std::cout<<"entering SiPixelActionExecutor::createEfficiency..."<<std::endl;
  bei->cd();
  fillEfficiency(bei, true, isUpgrade); // Barrel
  bei->cd();
  fillEfficiency(bei, false, isUpgrade); // Endcap
  bei->cd();
  //std::cout<<"leaving SiPixelActionExecutor::createEfficiency..."<<std::endl;
}

//=============================================================================================================

void SiPixelActionExecutor::fillEfficiency(DQMStore* bei, bool isbarrel, bool isUpgrade){
  //cout<<"entering SiPixelActionExecutor::fillEfficiency..."<<std::endl;
  string currDir = bei->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  //cout<<"currDir= "<<currDir<< " , dname= "<<dname<<std::endl;
  
  if(Tier0Flag_){ // Offline	
    if(isbarrel && dname.find("Ladder_")!=string::npos){ 
      if (!isUpgrade) {
      vector<string> meVec = bei->getMEs();
      for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
        string full_path = currDir + "/" + (*it);

	if(full_path.find("missingMod_")!=string::npos){ // If we have missing hits ME
	  
	  //Get the MEs that contain missing and valid hits
	  MonitorElement * missing = bei->get(full_path);
	  if (!missing) continue;
	  string new_path = full_path.replace(full_path.find("missing"),7,"valid");
	  MonitorElement * valid = bei->get(new_path);
	  if (!valid) continue;
	  //int binx = 0; 
	  int biny = 0;
	  //get the ladder number
	  
	  if(dname.find("01")!=string::npos){ biny = 1;}else if(dname.find("02")!=string::npos){ biny = 2;}
	  else if(dname.find("03")!=string::npos){ biny = 3;}else if(dname.find("04")!=string::npos){ biny = 4;}
	  else if(dname.find("05")!=string::npos){ biny = 5;}else if(dname.find("06")!=string::npos){ biny = 6;}
	  else if(dname.find("07")!=string::npos){ biny = 7;}else if(dname.find("08")!=string::npos){ biny = 8;}
	  else if(dname.find("09")!=string::npos){ biny = 9;}else if(dname.find("10")!=string::npos){ biny = 10;}
	  else if(dname.find("11")!=string::npos){ biny = 11;}else if(dname.find("12")!=string::npos){ biny = 12;}
	  else if(dname.find("13")!=string::npos){ biny = 13;}else if(dname.find("14")!=string::npos){ biny = 14;}
	  else if(dname.find("15")!=string::npos){ biny = 15;}else if(dname.find("16")!=string::npos){ biny = 16;}
	  else if(dname.find("17")!=string::npos){ biny = 17;}else if(dname.find("18")!=string::npos){ biny = 18;}
	  else if(dname.find("19")!=string::npos){ biny = 19;}else if(dname.find("20")!=string::npos){ biny = 20;}
	  else if(dname.find("21")!=string::npos){ biny = 21;}else if(dname.find("22")!=string::npos){ biny = 22;}
	  
	  if(currDir.find("Shell_mO")!=string::npos || currDir.find("Shell_pO")!=string::npos){
	    if(currDir.find("Layer_1")!=string::npos){ biny = biny + 10;}
	    else if(currDir.find("Layer_2")!=string::npos){ biny = biny + 16;}
	    
	    else if(currDir.find("Layer_3")!=string::npos){ biny = biny + 22;}
	    
	  }
	  
	  
	  
	  int start=1;
	  //define start depending on p or m
	  
	  if(currDir.find("Shell_m")!=string::npos){ start = 1;}else{ start = 5;}
	  for(int i=start; i<start+5;i++){
	    float hitEfficiency = -1.0;
	    float missingHits=0;
	    float validHits=0;
	    missingHits=missing->getBinContent(i);
	    validHits=valid->getBinContent(i);
	    if(validHits + missingHits > 0.) hitEfficiency = validHits / (validHits + missingHits);
	    if(currDir.find("Layer_1")!=string::npos){
	      HitEfficiency_L1 = bei->get("Pixel/Barrel/HitEfficiency_L1");
	      if(HitEfficiency_L1) HitEfficiency_L1->setBinContent(i, biny,(float)hitEfficiency);
	    }
	    else if(currDir.find("Layer_2")!=string::npos){
	      HitEfficiency_L2 = bei->get("Pixel/Barrel/HitEfficiency_L2");
	      if(HitEfficiency_L2) HitEfficiency_L2->setBinContent(i, biny,(float)hitEfficiency);
	    }
	    else if(currDir.find("Layer_3")!=string::npos){
	      HitEfficiency_L3 = bei->get("Pixel/Barrel/HitEfficiency_L3");
	      if(HitEfficiency_L3) HitEfficiency_L3->setBinContent(i, biny,(float)hitEfficiency);
	    }     
	    
	  }
	  
	}
      }
      }//endifNOTUpgradeInBPix
      else if (isUpgrade) {
        vector<string> meVec = bei->getMEs();
        for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
          string full_path = currDir + "/" + (*it);
          if(full_path.find("missing_")!=string::npos){ // If we have missing hits ME
	    MonitorElement * me = bei->get(full_path);
	    if (!me) continue;
	    float missingHits = me->getEntries();
	    //if(currDir.find("Barrel/Shell_mI/Layer_1/Ladder_09F")!=string::npos) cout<<"missingHits= "<<missingHits<<endl;
	    string new_path = full_path.replace(full_path.find("missing"),7,"valid");
	    me = bei->get(new_path);
	    if (!me) continue;
	    float validHits = me->getEntries();
	    //if(currDir.find("Barrel/Shell_mI/Layer_1/Ladder_09F")!=string::npos) cout<<"validHits= "<<validHits<<endl;
	    float hitEfficiency = -1.;
	    if(validHits + missingHits > 0.) hitEfficiency = validHits / (validHits + missingHits);
	    //if(currDir.find("Barrel/Shell_mI/Layer_1/Ladder_09F")!=string::npos) cout<<"hitEfficiency= "<<hitEfficiency<<endl;
	    int binx = 0; int biny = 0;
	    if(currDir.find("Shell_m")!=string::npos){ binx = 1;}else{ binx = 2;}
	    if(dname.find("01")!=string::npos){ biny = 1;}else if(dname.find("02")!=string::npos){ biny = 2;}
	    else if(dname.find("03")!=string::npos){ biny = 3;}else if(dname.find("04")!=string::npos){ biny = 4;}
	    else if(dname.find("05")!=string::npos){ biny = 5;}else if(dname.find("06")!=string::npos){ biny = 6;}
	    else if(dname.find("07")!=string::npos){ biny = 7;}else if(dname.find("08")!=string::npos){ biny = 8;}
	    else if(dname.find("09")!=string::npos){ biny = 9;}else if(dname.find("10")!=string::npos){ biny = 10;}
	    else if(dname.find("11")!=string::npos){ biny = 11;}else if(dname.find("12")!=string::npos){ biny = 12;}
	    else if(dname.find("13")!=string::npos){ biny = 13;}else if(dname.find("14")!=string::npos){ biny = 14;}
	    else if(dname.find("15")!=string::npos){ biny = 15;}else if(dname.find("16")!=string::npos){ biny = 16;}
	    else if(dname.find("17")!=string::npos){ biny = 17;}else if(dname.find("18")!=string::npos){ biny = 18;}
	    else if(dname.find("19")!=string::npos){ biny = 19;}else if(dname.find("20")!=string::npos){ biny = 20;}
	    else if(dname.find("21")!=string::npos){ biny = 21;}else if(dname.find("22")!=string::npos){ biny = 22;}
	    else if(dname.find("23")!=string::npos){ biny = 23;}else if(dname.find("24")!=string::npos){ biny = 24;}
	    else if(dname.find("25")!=string::npos){ biny = 25;}else if(dname.find("25")!=string::npos){ biny = 25;}
	    else if(dname.find("26")!=string::npos){ biny = 26;}else if(dname.find("27")!=string::npos){ biny = 27;}
	    else if(dname.find("28")!=string::npos){ biny = 28;}else if(dname.find("29")!=string::npos){ biny = 29;}
	    else if(dname.find("30")!=string::npos){ biny = 30;}else if(dname.find("31")!=string::npos){ biny = 31;}
	    else if(dname.find("32")!=string::npos){ biny = 32;}
	    if(currDir.find("Shell_mO")!=string::npos || currDir.find("Shell_pO")!=string::npos){
	      if(currDir.find("Layer_1")!=string::npos){ biny = biny + 6;}
	      else if(currDir.find("Layer_2")!=string::npos){ biny = biny + 14;}
	      else if(currDir.find("Layer_3")!=string::npos){ biny = biny + 22;}
	      else if(currDir.find("Layer_4")!=string::npos){ biny = biny + 32;}
	    }
	    if(currDir.find("Layer_1")!=string::npos){
	      HitEfficiency_L1 = bei->get("Pixel/Barrel/HitEfficiency_L1");
	      if(HitEfficiency_L1) HitEfficiency_L1->setBinContent(binx, biny,(float)hitEfficiency);
	      //if(currDir.find("Barrel/Shell_mI/Layer_1/Ladder_09F")!=string::npos) cout<<"setting bin ("<<binx<<","<<biny<<") with "<<(float)hitEfficiency<<endl;
	    }else if(currDir.find("Layer_2")!=string::npos){
	      HitEfficiency_L2 = bei->get("Pixel/Barrel/HitEfficiency_L2");
	      if(HitEfficiency_L2) HitEfficiency_L2->setBinContent(binx, biny,(float)hitEfficiency);
	    }else if(currDir.find("Layer_3")!=string::npos){
	      HitEfficiency_L3 = bei->get("Pixel/Barrel/HitEfficiency_L3");
	      if(HitEfficiency_L3) HitEfficiency_L3->setBinContent(binx, biny,(float)hitEfficiency);
	    }else if(currDir.find("Layer_4")!=string::npos){
	      HitEfficiency_L4 = bei->get("Pixel/Barrel/HitEfficiency_L4");
	      if(HitEfficiency_L4) HitEfficiency_L4->setBinContent(binx, biny,(float)hitEfficiency);
	    }
          }
        }
      }//endifUpgradeInBPix
    }else if(!isbarrel && dname.find("Blade_")!=string::npos && !isUpgrade){ 
      vector<string> meVec = bei->getMEs();
      for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
        string full_path = currDir + "/" + (*it);
        if(full_path.find("missing_")!=string::npos){ // If we have missing hits ME
	  MonitorElement * me = bei->get(full_path);
	  if (!me) continue;
	  float missingHits = me->getEntries();
	  string new_path = full_path.replace(full_path.find("missing"),7,"valid");
	  me = bei->get(new_path);
	  if (!me) continue;
	  float validHits = me->getEntries();
	  float hitEfficiency = -1.;
	  if(validHits + missingHits > 0.) hitEfficiency = validHits / (validHits + missingHits);
	  int binx = 0; int biny = 1;
	  if(currDir.find("01")!=string::npos){ binx = 1;}else if(currDir.find("02")!=string::npos){ binx = 2;}
	  else if(currDir.find("03")!=string::npos){ binx = 3;}else if(currDir.find("04")!=string::npos){ binx = 4;}
	  else if(currDir.find("05")!=string::npos){ binx = 5;}else if(currDir.find("06")!=string::npos){ binx = 6;}
	  else if(currDir.find("07")!=string::npos){ binx = 7;}else if(currDir.find("08")!=string::npos){ binx = 8;}
	  else if(currDir.find("09")!=string::npos){ binx = 9;}else if(currDir.find("10")!=string::npos){ binx = 10;}
	  else if(currDir.find("11")!=string::npos){ binx = 11;}else if(currDir.find("12")!=string::npos){ binx = 12;}
	  if(currDir.find("HalfCylinder_mI")!=string::npos || currDir.find("HalfCylinder_pI")!=string::npos){ binx = binx + 12;}
	  else{ 
	    if(binx==1) binx = 12;
	    else if(binx==2) binx = 11;
	    else if(binx==3) binx = 10;
	    else if(binx==4) binx = 9;
	    else if(binx==5) binx = 8;
	    else if(binx==6) binx = 7;
	    else if(binx==7) binx = 6;
	    else if(binx==8) binx = 5;
	    else if(binx==9) binx = 4;
	    else if(binx==10) binx = 3;
	    else if(binx==11) binx = 2;
	    else if(binx==12) binx = 1;
	  }
	  if(currDir.find("Disk_1")!=string::npos && currDir.find("HalfCylinder_m")!=string::npos){
	    HitEfficiency_Dm1 = bei->get("Pixel/Endcap/HitEfficiency_Dm1");
	    if(HitEfficiency_Dm1) HitEfficiency_Dm1->setBinContent(binx, biny, (float)hitEfficiency);
	  }else if(currDir.find("Disk_2")!=string::npos && currDir.find("HalfCylinder_m")!=string::npos){
	    HitEfficiency_Dm2 = bei->get("Pixel/Endcap/HitEfficiency_Dm2");
	    if(HitEfficiency_Dm2) HitEfficiency_Dm2->setBinContent(binx, biny, (float)hitEfficiency);
	  }else if(currDir.find("Disk_1")!=string::npos && currDir.find("HalfCylinder_p")!=string::npos){
	    HitEfficiency_Dp1 = bei->get("Pixel/Endcap/HitEfficiency_Dp1");
	    if(HitEfficiency_Dp1) HitEfficiency_Dp1->setBinContent(binx, biny, (float)hitEfficiency);
	  }else if(currDir.find("Disk_2")!=string::npos && currDir.find("HalfCylinder_p")!=string::npos){
	    HitEfficiency_Dp2 = bei->get("Pixel/Endcap/HitEfficiency_Dp2");
	    if(HitEfficiency_Dp2) HitEfficiency_Dp2->setBinContent(binx, biny, (float)hitEfficiency);
          }
	  //std::cout<<"EFFI: "<<currDir<<" , x: "<<binx<<" , y: "<<biny<<std::endl;
	}
      } 
    }else if(!isbarrel && dname.find("Blade_")!=string::npos && isUpgrade){ 
      vector<string> meVec = bei->getMEs();
      for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
        string full_path = currDir + "/" + (*it);
        if(full_path.find("missing_")!=string::npos){ // If we have missing hits ME
	  MonitorElement * me = bei->get(full_path);
	  if (!me) continue;
	  float missingHits = me->getEntries();
	  string new_path = full_path.replace(full_path.find("missing"),7,"valid");
	  me = bei->get(new_path);
	  if (!me) continue;
	  float validHits = me->getEntries();
	  float hitEfficiency = -1.;
	  if(validHits + missingHits > 0.) hitEfficiency = validHits / (validHits + missingHits);
	  int binx = 0; int biny = 1;
	  if(currDir.find("01")!=string::npos){ binx = 1;}else if(currDir.find("02")!=string::npos){ binx = 2;}
	  else if(currDir.find("03")!=string::npos){ binx = 3;}else if(currDir.find("04")!=string::npos){ binx = 4;}
	  else if(currDir.find("05")!=string::npos){ binx = 5;}else if(currDir.find("06")!=string::npos){ binx = 6;}
	  else if(currDir.find("07")!=string::npos){ binx = 7;}else if(currDir.find("08")!=string::npos){ binx = 8;}
	  else if(currDir.find("09")!=string::npos){ binx = 9;}else if(currDir.find("10")!=string::npos){ binx = 10;}
	  else if(currDir.find("11")!=string::npos){ binx = 11;}else if(currDir.find("12")!=string::npos){ binx = 12;}
          else if(currDir.find("13")!=string::npos){ binx = 13;}else if(currDir.find("14")!=string::npos){ binx = 14;}
          else if(currDir.find("15")!=string::npos){ binx = 15;}else if(currDir.find("16")!=string::npos){ binx = 16;}
          else if(currDir.find("17")!=string::npos){ binx = 17;}
	  if(currDir.find("HalfCylinder_mI")!=string::npos || currDir.find("HalfCylinder_pI")!=string::npos){ binx = binx + 12;}
	  else{ 
	    if(binx==1) binx = 17;
	    else if(binx==2) binx = 16;
	    else if(binx==3) binx = 15;
	    else if(binx==4) binx = 14;
	    else if(binx==5) binx = 13;
	    else if(binx==6) binx = 12;
	    else if(binx==7) binx = 11;
	    else if(binx==8) binx = 10;
	    else if(binx==9) binx = 9;
	    else if(binx==10) binx = 8;
	    else if(binx==11) binx = 7;
	    else if(binx==12) binx = 6;
            else if(binx==13) binx = 5;
	    else if(binx==14) binx = 4;
	    else if(binx==15) binx = 3;
	    else if(binx==16) binx = 2;
	    else if(binx==17) binx = 1;
	  }
	  if(currDir.find("Disk_1")!=string::npos && currDir.find("HalfCylinder_m")!=string::npos){
	    HitEfficiency_Dm1 = bei->get("Pixel/Endcap/HitEfficiency_Dm1");
	    if(HitEfficiency_Dm1) HitEfficiency_Dm1->setBinContent(binx, biny, (float)hitEfficiency);
	  }else if(currDir.find("Disk_2")!=string::npos && currDir.find("HalfCylinder_m")!=string::npos){
	    HitEfficiency_Dm2 = bei->get("Pixel/Endcap/HitEfficiency_Dm2");
	    if(HitEfficiency_Dm2) HitEfficiency_Dm2->setBinContent(binx, biny, (float)hitEfficiency);
	  }else if(currDir.find("Disk_3")!=string::npos && currDir.find("HalfCylinder_m")!=string::npos){
	    HitEfficiency_Dm3 = bei->get("Pixel/Endcap/HitEfficiency_Dm3");
	    if(HitEfficiency_Dm3) HitEfficiency_Dm3->setBinContent(binx, biny, (float)hitEfficiency);
	  }else if(currDir.find("Disk_1")!=string::npos && currDir.find("HalfCylinder_p")!=string::npos){
	    HitEfficiency_Dp1 = bei->get("Pixel/Endcap/HitEfficiency_Dp1");
	    if(HitEfficiency_Dp1) HitEfficiency_Dp1->setBinContent(binx, biny, (float)hitEfficiency);
	  }else if(currDir.find("Disk_2")!=string::npos && currDir.find("HalfCylinder_p")!=string::npos){
	    HitEfficiency_Dp2 = bei->get("Pixel/Endcap/HitEfficiency_Dp2");
	    if(HitEfficiency_Dp2) HitEfficiency_Dp2->setBinContent(binx, biny, (float)hitEfficiency);
          }else if(currDir.find("Disk_3")!=string::npos && currDir.find("HalfCylinder_p")!=string::npos){
	    HitEfficiency_Dp3 = bei->get("Pixel/Endcap/HitEfficiency_Dp3");
	    if(HitEfficiency_Dp3) HitEfficiency_Dp3->setBinContent(binx, biny, (float)hitEfficiency);
          }
	  //std::cout<<"EFFI: "<<currDir<<" , x: "<<binx<<" , y: "<<biny<<std::endl;
	}
      } 
    }else{  
      //cout<<"finding subdirs now"<<std::endl;
      vector<string> subdirs = bei->getSubdirs();
      for (vector<string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++) {
        bei->cd(*it);
        //cout<<"now I am in "<<bei->pwd()<<std::endl;
        if(*it != "Pixel" && ((isbarrel && (*it).find("Barrel")==string::npos) || (!isbarrel && (*it).find("Endcap")==string::npos))) continue;
        //cout<<"calling myself again "<<std::endl;
        fillEfficiency(bei, isbarrel, isUpgrade);
        bei->goUp();
      }
    }
  }else{ // Online
    if(dname.find("Module_")!=string::npos){ 
      vector<string> meVec = bei->getMEs();
      for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
        string full_path = currDir + "/" + (*it);
        if(full_path.find("missing_")!=string::npos){ // If we have missing hits ME
	  MonitorElement * me = bei->get(full_path);
	  if (!me) continue;
	  float missingHits = me->getEntries();
	  string new_path = full_path.replace(full_path.find("missing"),7,"valid");
	  me = bei->get(new_path);
	  if (!me) continue;
	  float validHits = me->getEntries();
	  float hitEfficiency = -1.;
	  if(validHits + missingHits > 0.) hitEfficiency = validHits / (validHits + missingHits);
	  int binx = 0; int biny = 0;
	  if(isbarrel){
	    if(currDir.find("Shell_m")!=string::npos){
	      if(currDir.find("Module_4")!=string::npos){ binx = 1;}else if(currDir.find("Module_3")!=string::npos){ binx = 2;}
	      if(currDir.find("Module_2")!=string::npos){ binx = 3;}else if(currDir.find("Module_1")!=string::npos){ binx = 4;}
	    }else if(currDir.find("Shell_p")!=string::npos){
	      if(currDir.find("Module_1")!=string::npos){ binx = 5;}else if(currDir.find("Module_2")!=string::npos){ binx = 6;}
	      if(currDir.find("Module_3")!=string::npos){ binx = 7;}else if(currDir.find("Module_4")!=string::npos){ binx = 8;}
	    }
	    if (!isUpgrade) {
	    if(currDir.find("01")!=string::npos){ biny = 1;}else if(currDir.find("02")!=string::npos){ biny = 2;}
	    else if(currDir.find("03")!=string::npos){ biny = 3;}else if(currDir.find("04")!=string::npos){ biny = 4;}
	    else if(currDir.find("05")!=string::npos){ biny = 5;}else if(currDir.find("06")!=string::npos){ biny = 6;}
	    else if(currDir.find("07")!=string::npos){ biny = 7;}else if(currDir.find("08")!=string::npos){ biny = 8;}
	    else if(currDir.find("09")!=string::npos){ biny = 9;}else if(currDir.find("10")!=string::npos){ biny = 10;}
	    else if(currDir.find("11")!=string::npos){ biny = 11;}else if(currDir.find("12")!=string::npos){ biny = 12;}
	    else if(currDir.find("13")!=string::npos){ biny = 13;}else if(currDir.find("14")!=string::npos){ biny = 14;}
	    else if(currDir.find("15")!=string::npos){ biny = 15;}else if(currDir.find("16")!=string::npos){ biny = 16;}
	    else if(currDir.find("17")!=string::npos){ biny = 17;}else if(currDir.find("18")!=string::npos){ biny = 18;}
	    else if(currDir.find("19")!=string::npos){ biny = 19;}else if(currDir.find("20")!=string::npos){ biny = 20;}
	    else if(currDir.find("21")!=string::npos){ biny = 21;}else if(currDir.find("22")!=string::npos){ biny = 22;}
	    if(currDir.find("Shell_mO")!=string::npos || currDir.find("Shell_pO")!=string::npos){
	      if(currDir.find("Layer_1")!=string::npos){ biny = biny + 10;}
	      else if(currDir.find("Layer_2")!=string::npos){ biny = biny + 16;}
	      else if(currDir.find("Layer_3")!=string::npos){ biny = biny + 22;}
	    }
	    }
	    else if (isUpgrade) {
	      if(currDir.find("01")!=string::npos){ biny = 1;}else if(currDir.find("02")!=string::npos){ biny = 2;}
	      else if(currDir.find("03")!=string::npos){ biny = 3;}else if(currDir.find("04")!=string::npos){ biny = 4;}
	      else if(currDir.find("05")!=string::npos){ biny = 5;}else if(currDir.find("06")!=string::npos){ biny = 6;}
	      else if(currDir.find("07")!=string::npos){ biny = 7;}else if(currDir.find("08")!=string::npos){ biny = 8;}
	      else if(currDir.find("09")!=string::npos){ biny = 9;}else if(currDir.find("10")!=string::npos){ biny = 10;}
	      else if(currDir.find("11")!=string::npos){ biny = 11;}else if(currDir.find("12")!=string::npos){ biny = 12;}
	      else if(currDir.find("13")!=string::npos){ biny = 13;}else if(currDir.find("14")!=string::npos){ biny = 14;}
	      else if(currDir.find("15")!=string::npos){ biny = 15;}else if(currDir.find("16")!=string::npos){ biny = 16;}
	      else if(currDir.find("17")!=string::npos){ biny = 17;}else if(currDir.find("18")!=string::npos){ biny = 18;}
	      else if(currDir.find("19")!=string::npos){ biny = 19;}else if(currDir.find("20")!=string::npos){ biny = 20;}
	      else if(currDir.find("21")!=string::npos){ biny = 21;}else if(currDir.find("22")!=string::npos){ biny = 22;}
	      else if(currDir.find("23")!=string::npos){ biny = 23;}else if(currDir.find("24")!=string::npos){ biny = 24;}
	      else if(currDir.find("25")!=string::npos){ biny = 25;}else if(currDir.find("25")!=string::npos){ biny = 25;}
	      else if(currDir.find("26")!=string::npos){ biny = 26;}else if(currDir.find("27")!=string::npos){ biny = 27;}
	      else if(currDir.find("28")!=string::npos){ biny = 28;}else if(currDir.find("29")!=string::npos){ biny = 29;}
	      else if(currDir.find("30")!=string::npos){ biny = 30;}else if(currDir.find("31")!=string::npos){ biny = 31;}
	      else if(currDir.find("32")!=string::npos){ biny = 32;}
	      if(currDir.find("Shell_mO")!=string::npos || currDir.find("Shell_pO")!=string::npos){
	        if(currDir.find("Layer_1")!=string::npos){ biny = biny + 6;}
	        else if(currDir.find("Layer_2")!=string::npos){ biny = biny + 14;}
	        else if(currDir.find("Layer_3")!=string::npos){ biny = biny + 22;}
	        else if(currDir.find("Layer_4")!=string::npos){ biny = biny + 32;}
	      }
	    }
	  }else{ //endcap
	    if (!isUpgrade) {
	    if(currDir.find("01")!=string::npos){ binx = 1;}else if(currDir.find("02")!=string::npos){ binx = 2;}
	    else if(currDir.find("03")!=string::npos){ binx = 3;}else if(currDir.find("04")!=string::npos){ binx = 4;}
	    else if(currDir.find("05")!=string::npos){ binx = 5;}else if(currDir.find("06")!=string::npos){ binx = 6;}
	    else if(currDir.find("07")!=string::npos){ binx = 7;}else if(currDir.find("08")!=string::npos){ binx = 8;}
	    else if(currDir.find("09")!=string::npos){ binx = 9;}else if(currDir.find("10")!=string::npos){ binx = 10;}
	    else if(currDir.find("11")!=string::npos){ binx = 11;}else if(currDir.find("12")!=string::npos){ binx = 12;}
	    if(currDir.find("HalfCylinder_mO")!=string::npos || currDir.find("HalfCylinder_pO")!=string::npos){ binx = binx + 12;}
	    if(currDir.find("Panel_1/Module_1")!=string::npos){ biny = 1;}else if(currDir.find("Panel_2/Module_1")!=string::npos){ biny = 2;}
	    else if(currDir.find("Panel_1/Module_2")!=string::npos){ biny = 3;}else if(currDir.find("Panel_2/Module_2")!=string::npos){ biny = 4;}
	    else if(currDir.find("Panel_1/Module_3")!=string::npos){ biny = 5;}else if(currDir.find("Panel_2/Module_3")!=string::npos){ biny = 6;}
	    else if(currDir.find("Panel_1/Module_4")!=string::npos){ biny = 7;}
	    } else if (isUpgrade) {
	      if(currDir.find("01")!=string::npos){ binx = 1;}else if(currDir.find("02")!=string::npos){ binx = 2;}
	      else if(currDir.find("03")!=string::npos){ binx = 3;}else if(currDir.find("04")!=string::npos){ binx = 4;}
	      else if(currDir.find("05")!=string::npos){ binx = 5;}else if(currDir.find("06")!=string::npos){ binx = 6;}
	      else if(currDir.find("07")!=string::npos){ binx = 7;}else if(currDir.find("08")!=string::npos){ binx = 8;}
	      else if(currDir.find("09")!=string::npos){ binx = 9;}else if(currDir.find("10")!=string::npos){ binx = 10;}
	      else if(currDir.find("11")!=string::npos){ binx = 11;}else if(currDir.find("12")!=string::npos){ binx = 12;}
	      else if(currDir.find("13")!=string::npos){ binx = 13;}else if(currDir.find("14")!=string::npos){ binx = 14;}
	      else if(currDir.find("15")!=string::npos){ binx = 15;}else if(currDir.find("16")!=string::npos){ binx = 16;}
	      else if(currDir.find("17")!=string::npos){ binx = 17;}
	      if(currDir.find("HalfCylinder_mO")!=string::npos || currDir.find("HalfCylinder_pO")!=string::npos){ binx = binx + 17;}
	      if(currDir.find("Panel_1/Module_1")!=string::npos){ biny = 1;}else if(currDir.find("Panel_2/Module_1")!=string::npos){ biny = 2;}
	    }//endif(isUpgrade)
	  }
	  
	  if(currDir.find("Layer_1")!=string::npos){
	    HitEfficiency_L1 = bei->get("Pixel/Barrel/HitEfficiency_L1");
	    if(HitEfficiency_L1) HitEfficiency_L1->setBinContent(binx, biny,(float)hitEfficiency);
	  }else if(currDir.find("Layer_2")!=string::npos){
	    HitEfficiency_L2 = bei->get("Pixel/Barrel/HitEfficiency_L2");
	    if(HitEfficiency_L2) HitEfficiency_L2->setBinContent(binx, biny,(float)hitEfficiency);
	  }else if(currDir.find("Layer_3")!=string::npos){
	    HitEfficiency_L3 = bei->get("Pixel/Barrel/HitEfficiency_L3");
	    if(HitEfficiency_L3) HitEfficiency_L3->setBinContent(binx, biny,(float)hitEfficiency);
	  }else if( isUpgrade && (currDir.find("Layer_4")!=string::npos) ){
	    HitEfficiency_L4 = bei->get("Pixel/Barrel/HitEfficiency_L4");
	    if(HitEfficiency_L4) HitEfficiency_L4->setBinContent(binx, biny,(float)hitEfficiency);
	  }else if(currDir.find("Disk_1")!=string::npos && currDir.find("HalfCylinder_m")!=string::npos){
	    HitEfficiency_Dm1 = bei->get("Pixel/Endcap/HitEfficiency_Dm1");
	    if(HitEfficiency_Dm1) HitEfficiency_Dm1->setBinContent(binx, biny,(float)hitEfficiency);
	  }else if(currDir.find("Disk_2")!=string::npos && currDir.find("HalfCylinder_m")!=string::npos){
	    HitEfficiency_Dm2 = bei->get("Pixel/Endcap/HitEfficiency_Dm2");
	    if(HitEfficiency_Dm2) HitEfficiency_Dm2->setBinContent(binx, biny,(float)hitEfficiency);
	  }else if(currDir.find("Disk_3")!=string::npos && currDir.find("HalfCylinder_m")!=string::npos){
	    HitEfficiency_Dm3 = bei->get("Pixel/Endcap/HitEfficiency_Dm3");
	    if(HitEfficiency_Dm3) HitEfficiency_Dm3->setBinContent(binx, biny,(float)hitEfficiency);
	  }else if(currDir.find("Disk_1")!=string::npos && currDir.find("HalfCylinder_p")!=string::npos){
	    HitEfficiency_Dp1 = bei->get("Pixel/Endcap/HitEfficiency_Dp1");
	    if(HitEfficiency_Dp1) HitEfficiency_Dp1->setBinContent(binx, biny,(float)hitEfficiency);
	  }else if(currDir.find("Disk_2")!=string::npos && currDir.find("HalfCylinder_p")!=string::npos){
	    HitEfficiency_Dp2 = bei->get("Pixel/Endcap/HitEfficiency_Dp2");
	    if(HitEfficiency_Dp2) HitEfficiency_Dp2->setBinContent(binx, biny,(float)hitEfficiency);
	  }else if(currDir.find("Disk_3")!=string::npos && currDir.find("HalfCylinder_p")!=string::npos){
	    HitEfficiency_Dp3 = bei->get("Pixel/Endcap/HitEfficiency_Dp3");
	    if(HitEfficiency_Dp3) HitEfficiency_Dp3->setBinContent(binx, biny,(float)hitEfficiency);
          }
        }
      }
    }else{  
      //cout<<"finding subdirs now"<<std::endl;
      vector<string> subdirs = bei->getSubdirs();
      for (vector<string>::const_iterator it = subdirs.begin(); it != subdirs.end(); it++) {
        bei->cd(*it);
        //cout<<"now I am in "<<bei->pwd()<<std::endl;
        if(*it != "Pixel" && ((isbarrel && (*it).find("Barrel")==string::npos) || (!isbarrel && (*it).find("Endcap")==string::npos))) continue;
        //cout<<"calling myself again "<<std::endl;
        fillEfficiency(bei, isbarrel, isUpgrade);
        bei->goUp();
      }
    }
  } // end online/offline
	
  //cout<<"leaving SiPixelActionExecutor::fillEfficiency..."<<std::endl;
	
}
