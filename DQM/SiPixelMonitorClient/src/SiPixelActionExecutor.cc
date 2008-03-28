#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelInformationExtractor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelTrackerMapCreator.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <qstring.h>
#include <qregexp.h>

#include <iostream>
using namespace std;
//
// -- Constructor
// 
SiPixelActionExecutor::SiPixelActionExecutor() {
  edm::LogInfo("SiPixelActionExecutor") << 
    " Creating SiPixelActionExecutor " << "\n" ;
  configParser_ = 0;
  configWriter_ = 0;
  qtHandler_ = 0;  
  ndet_ = 0;
  //collationDone = false;
}
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
//
// -- Read Configuration File
//
void SiPixelActionExecutor::readConfiguration() {
  string localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
  if (configParser_ == 0) {
    configParser_ = new SiPixelConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
}
//
// -- Read Configuration File
//
bool SiPixelActionExecutor::readConfiguration(int& tkmap_freq, 
                                              int& sum_barrel_freq, 
                                              int& sum_endcap_freq, 
					      int& sum_grandbarrel_freq, 
					      int& sum_grandendcap_freq, 
					      int& message_limit_,
					      int& source_type_) {
//cout<<"Entering SiPixelActionExecutor::readConfiguration..."<<endl;
  string localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
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
//cout<<"...leaving SiPixelActionExecutor::readConfiguration..."<<endl;
  return true;
}
bool SiPixelActionExecutor::readConfiguration(int& tkmap_freq, int& summary_freq) {
//cout<<"Entering SiPixelActionExecutor::readConfiguration..."<<endl;
  string localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
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
//cout<<"...leaving SiPixelActionExecutor::readConfiguration..."<<endl;
  return true;
}
//=============================================================================================================
// -- Create Tracker Map
//
void SiPixelActionExecutor::createTkMap(DQMStore* bei, 
                                        string mEName,
					string theTKType) 
{
 
  SiPixelTrackerMapCreator tkmap_creator(mEName,theTKType);
  tkmap_creator.create(bei);
  
//   cout << ACYellow << ACBold 
//        << "[SiPixelActionExecutor::createTkMap()]"
//        << ACPlain
//        << " Tracker map created (type:" 
//        << theTKType
//        << ")"
//        << endl;
}

//=============================================================================================================
void SiPixelActionExecutor::createSummary(DQMStore* bei) {
cout<<"entering SiPixelActionExecutor::createSummary..."<<endl;
  string barrel_structure_name;
  vector<string> barrel_me_names;
  if (!configParser_->getMENamesForBarrelSummary(barrel_structure_name, barrel_me_names)){
    cout << "SiPixelActionExecutor::createSummary: Failed to read Barrel Summary configuration parameters!! ";
    return;
  }
  configParser_->getSourceType(source_type_); 
  cout<<"Found source_type_="<<source_type_<<endl;
  bei->cd();
  fillBarrelSummary(bei, barrel_structure_name, barrel_me_names);
  bei->cd();
  string endcap_structure_name;
  vector<string> endcap_me_names;
  if (!configParser_->getMENamesForEndcapSummary(endcap_structure_name, endcap_me_names)){
    edm::LogInfo("SiPixelActionExecutor")  << "Failed to read Endcap Summary configuration parameters!! " << "\n" ;
    return;
  }
  bei->cd();
  fillEndcapSummary(bei, endcap_structure_name, endcap_me_names);
  
  bei->cd();
  if(source_type_==0||source_type_==5){//do this only if RawData source is present
    string federror_structure_name;
    vector<string> federror_me_names;
    if (!configParser_->getMENamesForFEDErrorSummary(federror_structure_name, federror_me_names)){
      cout << "SiPixelActionExecutor::createSummary: Failed to read FED Error Summary configuration parameters!! ";
      return;
    }
    bei->cd();
    fillFEDErrorSummary(bei, federror_structure_name, federror_me_names);
    bei->cd();
  }
  createLayout(bei);
  string fname = "test.xml";
  configWriter_->write(fname);
  if (configWriter_) delete configWriter_;
  configWriter_ = 0;
cout<<"leaving SiPixelActionExecutor::createSummary..."<<endl;
}


void SiPixelActionExecutor::fillBarrelSummary(DQMStore* bei,
                                              string dir_name,
					      vector<string>& me_names) {
  //cout<<"entering SiPixelActionExecutor::fillBarrelSummary..."<<endl;
  string currDir = bei->pwd();
  string prefix;
  if(source_type_==0) prefix="SUMRAW";
  else if (source_type_==1) prefix="SUMDIG";
  else if (source_type_==2) prefix="SUMCLU";
  else if (source_type_==3) prefix="SUMRES";
  else if (source_type_==4) prefix="SUMHIT";
  else if (source_type_>=7) prefix="SUMCAL";
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
	else if((*iv)=="residualX"||(*iv)=="residualY")
          prefix="SUMRES";
	else if((*iv)=="ClusX"||(*iv)=="ClusY")
	  prefix="SUMHIT";
	else if((*iv)=="Gain1d"||(*iv)=="GainChi2NDF1d"||
	   (*iv)=="GainChi2Prob1d"||(*iv)=="Pedestal1d"||
	   (*iv)=="ScurveChi2NDFSummary"||(*iv)=="ScurveFitResultSummary"||
	   (*iv)=="ScurveSigmasSummary"||(*iv)=="ScurveThresholdSummary"||
	   (*iv)=="pixelAliveSummary")
	  prefix="SUMCAL"; 
      }
      if((*iv).find("residual")!=string::npos ||        // track residuals
         prefix == "SUMCAL"){                           // calibrations
        string tag = prefix + "_" + (*iv) + "_mean_" 
                                + currDir.substr(currDir.find(dir_name));
        MonitorElement* temp = getSummaryME(bei, tag);
        sum_mes.push_back(temp);
        if((*iv)!="pixelAliveSummary"){
	  tag = prefix + "_" + (*iv) + "_RMS_" 
                                + currDir.substr(currDir.find(dir_name));
	}else{
	  tag = prefix + "_" + (*iv) + "_FracOfPerfectPix_"
                                + currDir.substr(currDir.find(dir_name));
	}
        temp = getSummaryME(bei, tag);
        sum_mes.push_back(temp);
      }else{
        string tag = prefix + "_" + (*iv) + "_" 
                                + currDir.substr(currDir.find(dir_name));
        MonitorElement* temp = getSummaryME(bei, tag);
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
      if ( (*it).find("Module_") == string::npos) continue;
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
	  if (((*im)).find(tname) == 0) {
	    string fullpathname = bei->pwd() + "/" + (*im); 

	    MonitorElement *  me = bei->get(fullpathname);
	    
	    if (me){ 
	      if (sname.find("_RMS_")!=string::npos){
	        (*isum)->Fill(ndet, me->getRMS());
	      }else if (sname.find("_FracOfPerfectPix_")!=string::npos){
	        cout<<"nbins = "<<me->getNbinsX()<<" , "<<me->getBinEntries(me->getNbinsX()-1)<<" , "<<me->getBinEntries(me->getNbinsX())<<endl;
		float nlast = me->getBinEntries(me->getNbinsX());
		float nall = me->getEntries();
	        (*isum)->Fill(ndet, nlast/nall);
              }else{
	        (*isum)->Fill(ndet, me->getMean());
	      }
              (*isum)->setAxisTitle("modules",1);
	      string title = " ";
	      if (sname.find("_RMS_")!=string::npos){
                title = "RMS of " + sname.substr(7,(sname.find("_",7)-7)) + " per module"; 
	      }else if (sname.find("_FracOfPerfectPix_")!=string::npos){
                title = "FracOfPerfectPix " + sname.substr(7,(sname.find("_",7)-7)) + " per module"; 
              }else{
                title = "Mean " + sname.substr(7,(sname.find("_",7)-7)) + " per module"; 
	      }
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
      if((bei->pwd()).find("Endcap")!=string::npos ||
         (bei->pwd()).find("AdditionalPixelErrors")!=string::npos) bei->goUp();
      bei->cd(*it);
      if((*it).find("Endcap")!=string::npos ||
         (*it).find("AdditionalPixelErrors")!=string::npos) continue;
      fillBarrelSummary(bei, dir_name, me_names);
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
 // cout<<"...leaving SiPixelActionExecutor::fillBarrelSummary!"<<endl;
}

void SiPixelActionExecutor::fillEndcapSummary(DQMStore* bei,
                                              string dir_name,
					      vector<string>& me_names) {
  //cout<<"entering SiPixelActionExecutor::fillEndcapSummary..."<<endl;
  string currDir = bei->pwd();
  string prefix;
  if(source_type_==0) prefix="SUMRAW";
  else if (source_type_==1) prefix="SUMDIG";
  else if (source_type_==2) prefix="SUMCLU";
  else if (source_type_==3) prefix="SUMRES";
  else if (source_type_==4) prefix="SUMHIT";
  else if (source_type_>=7) prefix="SUMCAL";
  
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
	else if((*iv)=="residualX"||(*iv)=="residualY")
          prefix="SUMRES";
	else if((*iv)=="ClusX"||(*iv)=="ClusY")
	  prefix="SUMHIT";
	else if((*iv)=="Gain1d"||(*iv)=="GainChi2NDF1d"||
	   (*iv)=="GainChi2Prob1d"||(*iv)=="Pedestal1d"||
	   (*iv)=="ScurveChi2NDFSummary"||(*iv)=="ScurveFitResultSummary"||
	   (*iv)=="ScurveSigmasSummary"||(*iv)=="ScurveThresholdSummary"||
	   (*iv)=="pixelAliveSummary")
	  prefix="SUMCAL"; 
      }
      if((*iv).find("residual")!=string::npos ||            // track residuals
         prefix == "SUMCAL"){                               // calibrations
        string tag = prefix + "_" + (*iv) + "_mean_" 
                                + currDir.substr(currDir.find(dir_name));
        MonitorElement* temp = getSummaryME(bei, tag);
        sum_mes.push_back(temp);
        if((*iv)!="pixelAliveSummary"){
	  tag = prefix + "_" + (*iv) + "_RMS_" 
                                + currDir.substr(currDir.find(dir_name));
	}else{
	  tag = prefix + "_" + (*iv) + "_FracOfPerfectPix_"
                                + currDir.substr(currDir.find(dir_name));
	}
        temp = getSummaryME(bei, tag);
        sum_mes.push_back(temp);
      }else{
        string tag = prefix + "_" + (*iv) + "_" 
                                + currDir.substr(currDir.find(dir_name));
	MonitorElement* temp = getSummaryME(bei, tag);
        sum_mes.push_back(temp);
      }
    }
    if (sum_mes.size() == 0) {
      edm::LogInfo("SiPixelActionExecutor")  << " Summary MEs can not be created" << "\n" ;
      return;
    }
    vector<string> subdirs = bei->getSubdirs();
    int ndet = 0;
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if ( (*it).find("Module_") == string::npos) continue;
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
	  if (((*im)).find(tname) == 0) {
	    string fullpathname = bei->pwd() + "/" + (*im); 

	    MonitorElement *  me = bei->get(fullpathname);
	    
	    if (me){ 
	      if (sname.find("_RMS_")!=string::npos){
	        (*isum)->Fill(ndet, me->getRMS());
	      }else if (sname.find("_FracOfPerfectPix_")!=string::npos){
		float nlast = me->getBinEntries(me->getNbinsX());
		float nall = me->getEntries();
	        (*isum)->Fill(ndet, nlast/nall);
              }else{
	        (*isum)->Fill(ndet, me->getMean());
	      }
              (*isum)->setAxisTitle("modules",1);
	      string title = " ";
	      if (sname.find("_RMS_")!=string::npos){
                title = "RMS of " + sname.substr(7,(sname.find("_",7)-7)) + " per module"; 
	      }else if (sname.find("_FracOfPerfectPix_")!=string::npos){
                title = "FracOfPerfectPix " + sname.substr(7,(sname.find("_",7)-7)) + " per module"; 
              }else{
                title = "Mean " + sname.substr(7,(sname.find("_",7)-7)) + " per module"; 
	      }
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
      if((bei->pwd()).find("Barrel")!=string::npos ||
         (bei->pwd()).find("AdditionalPixelErrors")!=string::npos) bei->goUp();
      bei->cd((*it));
      if ((*it).find("Barrel")!=string::npos ||
          (*it).find("AdditionalPixelErrors")!=string::npos) continue;
      fillEndcapSummary(bei, dir_name, me_names);
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
 // cout<<"...leaving SiPixelActionExecutor::fillEndcapSummary!"<<endl;
}


void SiPixelActionExecutor::fillFEDErrorSummary(DQMStore* bei,
                                                string dir_name,
						vector<string>& me_names) {
  //cout<<"entering SiPixelActionExecutor::fillFEDErrorSummary..."<<endl;
  string currDir = bei->pwd();
  string prefix;
  if(source_type_==0) prefix="SUMRAW";
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
      ndet++;

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
	      (*isum)->Fill(ndet, me->getMean());
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
  //cout<<"...leaving SiPixelActionExecutor::fillFEDErrorSummary!"<<endl;
}


void SiPixelActionExecutor::fillGrandBarrelSummaryHistos(DQMStore* bei,
                                                         vector<string>& me_names) {
//cout<<"Entering SiPixelActionExecutor::fillGrandBarrelSummaryHistos..."<<endl;
  vector<MonitorElement*> gsum_mes;
  string path_name = bei->pwd();
  string dir_name =  path_name.substr(path_name.find_last_of("/")+1);
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
    else if (source_type_==3) prefix="SUMRES";
    else if (source_type_==4) prefix="SUMHIT";
    else if (source_type_>=7) prefix="SUMCAL";
  
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
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
	  else if((*iv)=="residualX_mean"||(*iv)=="residualY_mean"||
	          (*iv)=="residualX_RMS"||(*iv)=="residualY_RMS")
            prefix="SUMRES";
	  else if((*iv)=="ClusX"||(*iv)=="ClusY")
	    prefix="SUMHIT";
	  else if((*iv)=="Gain1d_mean"||(*iv)=="GainChi2NDF1d_mean"||
	     (*iv)=="GainChi2Prob1d_mean"||(*iv)=="Pedestal1d_mean"||
	     (*iv)=="ScurveChi2NDFSummary_mean"||(*iv)=="ScurveFitResultSummary_mean"||
	     (*iv)=="ScurveSigmasSummary_mean"||(*iv)=="ScurveThresholdSummary_mean"||
	     (*iv)=="Gain1d_RMS"||(*iv)=="GainChi2NDF1d_RMS"||
	     (*iv)=="GainChi2Prob1d_RMS"||(*iv)=="Pedestal1d_RMS"||
	     (*iv)=="ScurveChi2NDFSummary_RMS"||(*iv)=="ScurveFitResultSummary_RMS"||
	     (*iv)=="ScurveSigmasSummary_RMS"||(*iv)=="ScurveThresholdSummary_RMS"||
	     (*iv)=="pixelAliveSummary_mean"||(*iv)=="pixelAliveSummary_FracOfPerfectPix")
	    prefix="SUMCAL";
        }
	string var = "_" + (*iv) + "_";
	if ((*im).find(var) != string::npos) {
	   string full_path = (*it) + "/" +(*im);
	   MonitorElement * me = bei->get(full_path.c_str());
	   
	   if (!me) continue; 
	   int actual_size = gsum_mes.size();
	   int wanted_size = me_names.size();
	   //if (dir_name=="Barrel") wanted_size = wanted_size * 2.;
           if (actual_size !=  wanted_size) {
	     nbin = me->getTH1F()->GetNbinsX();        
             string me_name = prefix + "_" + (*iv) + "_" + dir_name;
             if(dir_name=="Barrel") nbin=768;
	     else if(dir_name.find("Shell")!=string::npos) nbin=192;
	     else nbin=nbin*nDirs;
	     getGrandSummaryME(bei, nbin, me_name, gsum_mes);
	     //if (dir_name.find("Barrel")!=string::npos){
	     //  bei->goUp(); 
	     //  getGrandSummaryME(bei, nbin, me_name, gsum_mes);
	     //  bei->cd("Barrel");
	     //}
           }
	   for (vector<MonitorElement*>::const_iterator igm = gsum_mes.begin();
		igm != gsum_mes.end(); igm++) {
             if ((*igm)->getName().find(var) != string::npos) {
               (*igm)->setAxisTitle("modules",1);
               string title = "mean " + (*iv) + " per module"; 
               (*igm)->setAxisTitle(title,2);
	       if((*igm)->getName().find("Ladder") != string::npos){
		 nbin_i=0; nbin_subdir=4;
	       }else if((*igm)->getName().find("Layer") != string::npos){
		 nbin_i=(cnt-1)*4; nbin_subdir=4;
	       }else if((*igm)->getName().find("Shell") != string::npos){
	         if(iDir==0){ nbin_i=0; nbin_subdir=40; }
	         else if(iDir==1){ nbin_i=40; nbin_subdir=64; }
	         else if(iDir==2){ nbin_i=104; nbin_subdir=88; }
	       }else if((*igm)->getName().find("Barrel") != string::npos){
	         if(iDir==0){ nbin_i=0; nbin_subdir=192; }
		 else if(iDir==1){ nbin_i=192; nbin_subdir=192; }
		 else if(iDir==2){ nbin_i=384; nbin_subdir=192; }
		 else if(iDir==3){ nbin_i=576; nbin_subdir=192; }
	       }
	       for (int k = 1; k < nbin_subdir+1; k++) {
		 (*igm)->setBinContent(k+nbin_i, me->getBinContent(k));
	       }
             }
           }
	}
      }
    }
    iDir++;
  }
//cout<<"...leaving SiPixelActionExecutor::fillGrandBarrelSummaryHistos!"<<endl;
}

void SiPixelActionExecutor::fillGrandEndcapSummaryHistos(DQMStore* bei,
                                                         vector<string>& me_names) {
//cout<<"Entering SiPixelActionExecutor::fillGrandEndcapSummaryHistos..."<<endl;
  vector<MonitorElement*> gsum_mes;
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
  for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
    cnt++;
    bei->cd(*it);
    //cout<<"I am in "<<bei->pwd()<<" now!"<<endl;
    vector<string> contents = bei->getMEs();
   
    bei->goUp();
    
    string prefix;
    if(source_type_==0) prefix="SUMRAW";
    else if (source_type_==1) prefix="SUMDIG";
    else if (source_type_==2) prefix="SUMCLU";
    else if (source_type_==3) prefix="SUMRES";
    else if (source_type_==4) prefix="SUMHIT";
    else if (source_type_>=7) prefix="SUMCAL";
  
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
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
	  else if((*iv)=="residualX_mean"||(*iv)=="residualY_mean"||
	          (*iv)=="residualX_RMS"||(*iv)=="residualY_RMS")
            prefix="SUMRES";
	  else if((*iv)=="ClusX"||(*iv)=="ClusY")
	    prefix="SUMHIT";
	  else if((*iv)=="Gain1d_mean"||(*iv)=="GainChi2NDF1d_mean"||
	     (*iv)=="GainChi2Prob1d_mean"||(*iv)=="Pedestal1d_mean"||
	     (*iv)=="ScurveChi2NDFSummary_mean"||(*iv)=="ScurveFitResultSummary_mean"||
	     (*iv)=="ScurveSigmasSummary_mean"||(*iv)=="ScurveThresholdSummary_mean"||
	     (*iv)=="Gain1d_RMS"||(*iv)=="GainChi2NDF1d_RMS"||
	     (*iv)=="GainChi2Prob1d_RMS"||(*iv)=="Pedestal1d_RMS"||
	     (*iv)=="ScurveChi2NDFSummary_RMS"||(*iv)=="ScurveFitResultSummary_RMS"||
	     (*iv)=="ScurveSigmasSummary_RMS"||(*iv)=="ScurveThresholdSummary_RMS"||
	     (*iv)=="pixelAliveSummary_mean"||(*iv)=="pixelAliveSummary_FracOfPerfectPix")
	    prefix="SUMCAL"; 
        }
	string var = "_" + (*iv) + "_";
	//cout<<"prefix is "<<prefix<<" , and var is "<<var<<" , looking in "<<(*im)<<endl;
	if ((*im).find(var) != string::npos) {
	   string full_path = (*it) + "/" +(*im);

	   MonitorElement * me = bei->get(full_path.c_str());
	   
	   if (!me) continue; 
	   int actual_size = gsum_mes.size();
	   int wanted_size = me_names.size();
	   //cout<<"Found the ME and have "<<actual_size<<" , "<<wanted_size<<endl;
	   //if (dir_name=="Endcap") wanted_size = wanted_size * 2.;
           if (actual_size !=  wanted_size) {
	     nbin = me->getTH1F()->GetNbinsX();        
             string me_name = prefix + "_" + (*iv) + "_" + dir_name;
	     //cout<<"nbin for ME "<<me_name<<" is "<<nbin<<endl;
             if(dir_name=="Endcap") nbin=672;
	     else if(dir_name.find("HalfCylinder")!=string::npos) nbin=168;
	     else if(dir_name.find("Disk")!=string::npos) nbin=84;
	     else if(dir_name.find("Blade")!=string::npos) nbin=7;
	     else if(dir_name.find("Panel_1")!=string::npos) nbin=4;
	     else if(dir_name.find("Panel_2")!=string::npos) nbin=3;
	     //cout<<"now: me_name="<<me_name<<" , nbins="<<nbin<<" , creating histogram now!"<<endl;
	     getGrandSummaryME(bei, nbin, me_name, gsum_mes);
	     //if (dir_name.find("Endcap")!=string::npos){
	     //  bei->goUp(); 
	     //  getGrandSummaryME(bei, nbin, me_name, gsum_mes);
	     //  bei->cd("Endcap");
	     //}
	   }
	   for (vector<MonitorElement*>::const_iterator igm = gsum_mes.begin();
		igm != gsum_mes.end(); igm++) {
             if ((*igm)->getName().find(var) != string::npos) {
               (*igm)->setAxisTitle("modules",1);
               string title = "mean " + (*iv) + " per module"; 
               (*igm)->setAxisTitle(title,2);
	       nbin_i=0; 
	       if((*igm)->getName().find("Panel_1") != string::npos){
		 nbin_subdir=4;
	       }else if((*igm)->getName().find("Panel_2") != string::npos){
		 nbin_subdir=3;
	       }else if((*igm)->getName().find("Blade") != string::npos){
	         if((*im).find("_1") != string::npos) nbin_subdir=4;
	         if((*im).find("_2") != string::npos) {nbin_i=4; nbin_subdir=3;}
	       }else if((*igm)->getName().find("Disk") != string::npos){
	         nbin_i=((cnt-1)%12)*7; nbin_subdir=7;
	       }else if((*igm)->getName().find("HalfCylinder") != string::npos){
	         nbin_subdir=84;
	         if((*im).find("_2") != string::npos) nbin_i=84;
	       }else if((*igm)->getName().find("Endcap") != string::npos){
	         nbin_subdir=168;
	         if((*im).find("_mO") != string::npos) nbin_i=168;
	         if((*im).find("_pI") != string::npos) nbin_i=336;
	         if((*im).find("_pO") != string::npos) nbin_i=504;
	       }
	       for (int k = 1; k < nbin_subdir+1; k++) {
		 (*igm)->setBinContent(k+nbin_i, me->getBinContent(k));
	       }
             }
           }
	}
      }
    }
    iDir++;
  }
}
//
// -- Get Summary ME
//
void SiPixelActionExecutor::getGrandSummaryME(DQMStore* bei,
                                              int nbin, 
					      string& me_name, 
					      vector<MonitorElement*> & mes) {

  vector<string> contents = bei->getMEs();
      
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(me_name) == 0) {
      string fullpathname = bei->pwd() + "/" + me_name;

      MonitorElement* me = bei->get(fullpathname);
      
      if (me) {
      //cout<<"Found ME "<<fullpathname<<endl;
	me->Reset();
	mes.push_back(me);
	return;
      }
    }
  }
  MonitorElement* temp_me = bei->book1D(me_name.c_str(),me_name.c_str(),nbin,1.,nbin+1.);
  if (temp_me) mes.push_back(temp_me);
}


//
// -- Get Summary ME
//
MonitorElement* SiPixelActionExecutor::getSummaryME(DQMStore* bei,
                                                    string me_name) {
//cout<<"Entering SiPixelActionExecutor::getSummaryME..."<<endl;
  MonitorElement* me = 0;
  vector<string> contents = bei->getMEs();    
  
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(me_name) == 0) {
      string fullpathname = bei->pwd() + "/" + (*it); 

      me = bei->get(fullpathname);
      
      if (me) {
	me->Reset();
	return me;
      }
    }
  }
  me = bei->book1D(me_name.c_str(), me_name.c_str(),4,1.,5.);
  return me;
  //cout<<"...leaving SiPixelActionExecutor::getSummaryME!"<<endl;
}


MonitorElement* SiPixelActionExecutor::getFEDSummaryME(DQMStore* bei,
                                                       string me_name) {
//cout<<"Entering SiPixelActionExecutor::getFEDSummaryME..."<<endl;
  MonitorElement* me = 0;
  vector<string> contents = bei->getMEs();
      
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(me_name) == 0) {
      string fullpathname = bei->pwd() + "/" + (*it); 

      me = bei->get(fullpathname);
      
      if (me) {
	me->Reset();
	return me;
      }
    }
  }
  me = bei->book1D(me_name.c_str(), me_name.c_str(),40,-0.5,39.5);
  return me;
  //cout<<"...leaving SiPixelActionExecutor::getFEDSummaryME!"<<endl;
}
//
// -- Setup Quality Tests 
//
void SiPixelActionExecutor::setupQTests(DQMStore * bei) {
//cout<<"Entering SiPixelActionExecutor::setupQTests: "<<endl;

  bei->cd();
  bei->cd("Pixel");
  
  string localPath = string("DQM/SiPixelMonitorClient/test/sipixel_qualitytest_config.xml");
  if(!qtHandler_){
    qtHandler_ = new QTestHandle();
  }
  if(!qtHandler_->configureTests(edm::FileInPath(localPath).fullPath(),bei)){
    qtHandler_->attachTests(bei);
    bei->cd();
  }else{
    cout << " Problem setting up quality tests "<<endl;
  }

//cout<<" leaving SiPixelActionExecutor::setupQTests. "<<endl;
}
//
// -- Check Status of Quality Tests
//
void SiPixelActionExecutor::checkQTestResults(DQMStore * bei) {
//cout<<"Entering SiPixelActionExecutor::checkQTestResults..."<<endl;

  int messageCounter=0;
  string currDir = bei->pwd();
  vector<string> contentVec;
  bei->getContents(contentVec);
  
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
	  
	    cout <<  " *** Warning for " << me->getName() << "," 
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
	  messageCounter++;
	  if(messageCounter<message_limit_) {
	    //edm::LogError("SiPixelQualityTester::checkTestResults") << 
	    //  " *** Error for " << me->getName() << 
	    //  "," << (*ei)->getMessage() << "\n";
	  
	    cout  <<   " *** Error for " << me->getName() << ","
		  << (*ei)->getMessage() << " " << me->getMean() 
		  << " " << me->getRMS() 
		  << endl;
	  }
	}
	errors=vector<QReport*>();
      }
      me=0;
    }
    nval=int(); contents=vector<string>();
  }
  cout<<"messageCounter: "<<messageCounter<<" , message_limit: "<<message_limit_<<endl;
  if (messageCounter>=message_limit_)
    cout<<"WARNING: too many QTest failures! Giving up after "<<message_limit_<<" messages."<<endl;
  contentVec=vector<string>(); currDir=string(); messageCounter=int();
  //cout<<"...leaving SiPixelActionExecutor::checkQTestResults!"<<endl;
}

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

///// Dump Module paths and IDs on screen:
void SiPixelActionExecutor::dumpModIds(DQMStore * bei){
//cout<<"Going to dump module IDs now!"<<endl;
  bei->cd();
  dumpBarrelModIds(bei);
  bei->cd();
  dumpEndcapModIds(bei);
  bei->cd();
//cout<<"Done dumping module IDs!"<<endl;
}


void SiPixelActionExecutor::dumpBarrelModIds(DQMStore * bei){
  string currDir = bei->pwd();
  string dir_name = "Ladder_";
  if (currDir.find(dir_name) != string::npos)  {
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if ( (*it).find("Module_") == string::npos) continue;
      bei->cd(*it);
      ndet_++;
      cout<<"Ndet: "<<ndet_<<"  ,  Module: "<<bei->pwd();  
      vector<string> contents = bei->getMEs(); 
      bool first_me = false;
      int detId = -999;
      for (vector<string>::const_iterator im = contents.begin();
         im != contents.end(); im++) {
        if(first_me) break;
        QRegExp rx("(\\w+)_(\\w+)_(\\d+)") ;
        QString mEName = (*im);
        if(rx.search(mEName) != -1 ) detId = rx.cap(3).toInt() ;
      }
      bei->goUp();
      cout<<"  , detector ID: "<<detId<<endl;
    }
  } else {  
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if((*it).find("Endcap")!=string::npos) continue;
      bei->cd(*it);
      dumpBarrelModIds(bei);
      bei->goUp();
    }
  }
}

void SiPixelActionExecutor::dumpEndcapModIds(DQMStore * bei){
  string currDir = bei->pwd();
  string dir_name = "Panel_";
  if (currDir.find(dir_name) != string::npos)  {
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if ( (*it).find("Module_") == string::npos) continue;
      bei->cd(*it);
      ndet_++;
      cout<<"Ndet: "<<ndet_<<"  ,  Module: "<<bei->pwd();  
      vector<string> contents = bei->getMEs(); 
      bool first_me = false;
      int detId = -999;
      for (vector<string>::const_iterator im = contents.begin();
         im != contents.end(); im++) {
        if(first_me) break;
        QRegExp rx("(\\w+)_(\\w+)_(\\d+)") ;
        QString mEName = (*im);
        if(rx.search(mEName) != -1 ) detId = rx.cap(3).toInt() ;
      }
      bei->goUp();
      cout<<"  , detector ID: "<<detId<<endl;
    }
  } else {  
    vector<string> subdirs = bei->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if((bei->pwd()).find("Barrel")!=string::npos) bei->goUp();
      bei->cd((*it));
      if((*it).find("Barrel")!=string::npos) continue;
      dumpEndcapModIds(bei);
      bei->goUp();
    }
  }

}

