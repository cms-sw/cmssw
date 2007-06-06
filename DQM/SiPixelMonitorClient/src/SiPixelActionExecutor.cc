#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"
//#include "DQM/SiPixelMonitorClient/interface/SiPixelQualityTester.h"
#include "DQM/SiPixelMonitorClient/interface/TrackerMapCreator.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TText.h"
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
  collationDone = false;
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
bool SiPixelActionExecutor::readConfiguration(int& tkmap_freq, int& sum_barrel_freq, int& sum_endcap_freq) {
  string localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
  if (configParser_ == 0) {
    configParser_ = new SiPixelConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
//  if (!configParser_->getFrequencyForTrackerMap(tkmap_freq)){
//    cout << "SiPixelActionExecutor::readConfiguration: Failed to read TrackerMap configuration parameters!! ";
//    return false;
//  }
  if (!configParser_->getFrequencyForBarrelSummary(sum_barrel_freq)){
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read Barrel Summary configuration parameters!! " << "\n" ;
    return false;
  }
  if (!configParser_->getFrequencyForEndcapSummary(sum_endcap_freq)){
    edm::LogInfo("SiPixelActionExecutor")  << "Failed to read Endcap Summary configuration parameters!! " << "\n" ;
    return false;
  }
  return true;
}
//=============================================================================================================
// -- Create Tracker Map
//
void SiPixelActionExecutor::createTkMap(MonitorUserInterface* mui, string mEName) 
{
 
  TrackerMapCreator tkmap_creator(mEName);
  tkmap_creator.create(mui);
  
  cout << ACYellow << ACBold 
       << "[SiPixelActionExecutor::createTkMap()]"
       << ACPlain
       << " Tracker map created " << endl;
}

//=============================================================================================================
void SiPixelActionExecutor::createSummary(MonitorUserInterface* mui) {
//cout<<"entering SiPixelActionExecutor::createSummary..."<<endl;
  string barrel_structure_name;
  vector<string> barrel_me_names;
  if (!configParser_->getMENamesForBarrelSummary(barrel_structure_name, barrel_me_names)){
    cout << "SiPixelActionExecutor::createSummary: Failed to read Barrel Summary configuration parameters!! ";
    return;
  }
  mui->cd();
  fillBarrelSummary(mui, barrel_structure_name, barrel_me_names);
  mui->cd();
  string endcap_structure_name;
  vector<string> endcap_me_names;
  if (!configParser_->getMENamesForEndcapSummary(endcap_structure_name, endcap_me_names)){
    edm::LogInfo("SiPixelActionExecutor")  << "Failed to read Endcap Summary configuration parameters!! " << "\n" ;
    return;
  }
  mui->cd();
  fillEndcapSummary(mui, endcap_structure_name, endcap_me_names);
  mui->cd();
  createLayout(mui);
  string fname = "test.xml";
  configWriter_->write(fname);
  if (configWriter_) delete configWriter_;
  configWriter_ = 0;
//cout<<"leaving SiPixelActionExecutor::createSummary..."<<endl;
}
void SiPixelActionExecutor::fillBarrelSummary(MonitorUserInterface* mui,
                               string dir_name,vector<string>& me_names) {
  //cout<<"entering SiPixelActionExecutor::fillBarrelSummary..."<<endl;
  string currDir = mui->pwd();
  if (currDir.find(dir_name) != string::npos)  {
    vector<MonitorElement*> sum_mes;
    for (vector<string>::const_iterator iv = me_names.begin();
	 iv != me_names.end(); iv++) {
      string tag = "Summary_" + (*iv) + "_in_" 
                                + currDir.substr(currDir.find(dir_name));
      MonitorElement* temp = getSummaryME(mui, tag);
      sum_mes.push_back(temp);
    }
    if (sum_mes.size() == 0) {
      edm::LogInfo("SiPixelActionExecutor") << " Summary MEs can not be created" << "\n" ;
      return;
    }
    vector<string> subdirs = mui->getSubdirs();
    int ndet = 0;
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if ( (*it).find("Module_") == string::npos) continue;
      mui->cd(*it);
      ndet++;
      vector<string> contents = mui->getMEs(); 
      for (vector<MonitorElement*>::const_iterator isum = sum_mes.begin();
	   isum != sum_mes.end(); isum++) {
	for (vector<string>::const_iterator im = contents.begin();
	     im != contents.end(); im++) {
          string sname = ((*isum)->getName());
          string tname = sname.substr(8,(sname.find("_",8)-8));
	  if (((*im)).find(tname) == 0) {
	    string fullpathname = mui->pwd() + "/" + (*im); 
	    MonitorElement *  me = mui->get(fullpathname);
	    if (me){ 
	      (*isum)->Fill(ndet, me->getMean());
	    }
            break;
          }
	}
      }
      mui->goUp();
    }
  } else {  
    vector<string> subdirs = mui->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if((*it).find("PixelEndcap")!=string::npos) continue;
      mui->cd(*it);
      fillBarrelSummary(mui, dir_name, me_names);
      mui->goUp();
    }
    fillGrandBarrelSummaryHistos(mui, me_names);
  }
  //cout<<"...leaving SiPixelActionExecutor::fillBarrelSummary!"<<endl;
}

void SiPixelActionExecutor::fillEndcapSummary(MonitorUserInterface* mui,
                               string dir_name,vector<string>& me_names) {
  //cout<<"entering SiPixelActionExecutor::fillEndcapSummary..."<<endl;
  string currDir = mui->pwd();
  if (currDir.find(dir_name) != string::npos)  {
    vector<MonitorElement*> sum_mes;
    for (vector<string>::const_iterator iv = me_names.begin();
	 iv != me_names.end(); iv++) {
      string tag = "Summary_" + (*iv) + "_in_" 
                                + currDir.substr(currDir.find(dir_name));
      MonitorElement* temp = getSummaryME(mui, tag);
      sum_mes.push_back(temp);
    }
    if (sum_mes.size() == 0) {
      edm::LogInfo("SiPixelActionExecutor")  << " Summary MEs can not be created" << "\n" ;
      return;
    }
    vector<string> subdirs = mui->getSubdirs();
    int ndet = 0;
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if ( (*it).find("Module_") == string::npos) continue;
      mui->cd(*it);
      ndet++;
      vector<string> contents = mui->getMEs(); 
      for (vector<MonitorElement*>::const_iterator isum = sum_mes.begin();
	   isum != sum_mes.end(); isum++) {
	for (vector<string>::const_iterator im = contents.begin();
	     im != contents.end(); im++) {
          string sname = ((*isum)->getName());
          string tname = sname.substr(8,(sname.find("_",8)-8));
	  if (((*im)).find(tname) == 0) {
	    string fullpathname = mui->pwd() + "/" + (*im); 
	    MonitorElement *  me = mui->get(fullpathname);
	    if (me){ 
	      (*isum)->Fill(ndet, me->getMean());
	    }
            break;
          }
	}
      }
      mui->goUp();
    }
  } else {  
    vector<string> subdirs = mui->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if((mui->pwd()).find("PixelBarrel")!=string::npos) mui->goUp();
      mui->cd((*it));
      if ((*it).find("PixelBarrel")!=string::npos) continue;
      fillEndcapSummary(mui, dir_name, me_names);
      mui->goUp();
    }
    fillGrandEndcapSummaryHistos(mui, me_names);
  }
  //cout<<"...leaving SiPixelActionExecutor::fillEndcapSummary!"<<endl;
}
void SiPixelActionExecutor::fillGrandBarrelSummaryHistos(MonitorUserInterface* mui,
                                 vector<string>& me_names) {
//cout<<"Entering SiPixelActionExecutor::fillGrandBarrelSummaryHistos..."<<endl;
  vector<MonitorElement*> gsum_mes;
  string path_name = mui->pwd();
  string dir_name =  path_name.substr(path_name.find_last_of("/")+1);
  if ((dir_name.find("Collector") == 0) ||
      (dir_name.find("FU") == 0) ||
      (dir_name.find("Tracker") == 0) ||
      (dir_name.find("PixelEndcap") == 0) ||
      (dir_name.find("HalfCylinder") == 0) ||
      (dir_name.find("Disk") == 0) ||
      (dir_name.find("Blade") == 0) ||
      (dir_name.find("Panel") == 0) ) return;
  vector<string> subdirs = mui->getSubdirs();
  int nDirs = subdirs.size();
  int iDir =0;
  int nbin = 0;
  int nbin_i = 0; 
  int nbin_f = 0; 
  int cnt=0;
  for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
    cnt++;
    mui->cd(*it);
    vector<string> contents = mui->getMEs();
    mui->goUp();
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
      for (vector<string>::const_iterator iv = me_names.begin();
	   iv != me_names.end(); iv++) {
	string var = "_" + (*iv) + "_";
	if ((*im).find(var) != string::npos) {
	  string full_path = path_name + "/" + (*it) + "/" +(*im);
	   MonitorElement * me = mui->get(full_path.c_str());
	   if (!me) continue; 
           if (gsum_mes.size() !=  me_names.size()) {
	     MonitorElementT<TNamed>* obh1 = 
	       dynamic_cast<MonitorElementT<TNamed>*> (me);
	     if (obh1) {
	       TH1F * root_obh1 = dynamic_cast<TH1F *> (obh1->operator->());
	       if (root_obh1) nbin = root_obh1->GetNbinsX();        
	     } else nbin = 7777;
             string me_name = "Summary_" + (*iv) + "_in_" + dir_name;
             if(dir_name=="PixelBarrel") nbin=768;
	     else if(dir_name.find("Shell")!=string::npos) nbin=192;
	     else nbin=nbin*nDirs;
	     getGrandSummaryME(mui, nbin, me_name, gsum_mes);
           }
	   for (vector<MonitorElement*>::const_iterator igm = gsum_mes.begin();
		igm != gsum_mes.end(); igm++) {
             if ((*igm)->getName().find(var) != string::npos) {
	       if((*igm)->getName().find("Ladder") != string::npos){
		 nbin_i=0; nbin_f=4;
	       }else if((*igm)->getName().find("Layer") != string::npos){
		 nbin_i=(cnt-1)*4; nbin_f=40;
		 if(cnt>=11) nbin_f += 24;
		 if(cnt>=17) nbin_f += 24;
	       }else if((*igm)->getName().find("Shell") != string::npos){
	         if(iDir==0){ nbin_i=0; nbin_f=40; }
	         else if(iDir==1){ nbin_i=40; nbin_f=64; }
	         else if(iDir==2){ nbin_i=104; nbin_f=88; }
	       }else if((*igm)->getName().find("PixelBarrel") != string::npos){
	         if(iDir==0){ nbin_i=0; nbin_f=192; }
		 else if(iDir==1){ nbin_i=192; nbin_f=192; }
		 else if(iDir==2){ nbin_i=384; nbin_f=192; }
		 else if(iDir==3){ nbin_i=576; nbin_f=192; }
	       }
	       for (int k = 1; k < nbin_f+1; k++) {
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

void SiPixelActionExecutor::fillGrandEndcapSummaryHistos(MonitorUserInterface* mui,
                                 vector<string>& me_names) {
//cout<<"Entering SiPixelActionExecutor::fillGrandEndcapSummaryHistos..."<<endl;
  vector<MonitorElement*> gsum_mes;
  string path_name = mui->pwd();
  string dir_name =  path_name.substr(path_name.find_last_of("/")+1);
  if ((dir_name.find("Collector") == 0) ||
      (dir_name.find("FU") == 0) ||
      (dir_name.find("Tracker") == 0) ||
      (dir_name.find("PixelBarrel") == 0) ||
      (dir_name.find("Shell") == 0) ||
      (dir_name.find("Layer") == 0) ||
      (dir_name.find("Ladder") == 0) ) return;
  vector<string> subdirs = mui->getSubdirs();
  int iDir =0;
  int nbin = 0;
  int nbin_i = 0; 
  int nbin_f = 0; 
  int cnt=0;
  for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
    cnt++;
    mui->cd(*it);
    vector<string> contents = mui->getMEs();
    mui->goUp();
    
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
      for (vector<string>::const_iterator iv = me_names.begin();
	   iv != me_names.end(); iv++) {
	string var = "_" + (*iv) + "_";
	if ((*im).find(var) != string::npos) {
	  string full_path = path_name + "/" + (*it) + "/" +(*im);
	   MonitorElement * me = mui->get(full_path.c_str());
	   if (!me) continue; 
           if (gsum_mes.size() !=  me_names.size()) {
	     MonitorElementT<TNamed>* obh1 = 
	       dynamic_cast<MonitorElementT<TNamed>*> (me);
	     if (obh1) {
	       TH1F * root_obh1 = dynamic_cast<TH1F *> (obh1->operator->());
	       if (root_obh1) nbin = root_obh1->GetNbinsX();        
	     } else nbin = 7777;
             string me_name = "Summary_" + (*iv) + "_in_" + dir_name;
             if(dir_name=="PixelEndcap") nbin=672;
	     else if(dir_name.find("HalfCylinder")!=string::npos) nbin=168;
	     else if(dir_name.find("Disk")!=string::npos) nbin=84;
	     else if(dir_name.find("Blade")!=string::npos) nbin=7;
	     else if(dir_name.find("Panel_1")!=string::npos) nbin=4;
	     else if(dir_name.find("Panel_2")!=string::npos) nbin=3;
	     getGrandSummaryME(mui, nbin, me_name, gsum_mes);
	   }
	   for (vector<MonitorElement*>::const_iterator igm = gsum_mes.begin();
		igm != gsum_mes.end(); igm++) {
             if ((*igm)->getName().find(var) != string::npos) {
	       nbin_i=0; 
	       if((*igm)->getName().find("Panel_1") != string::npos){
		 nbin_f=4;
	       }else if((*igm)->getName().find("Panel_2") != string::npos){
		 nbin_f=3;
	       }else if((*igm)->getName().find("Blade") != string::npos){
	         nbin_f=7;
	         if((*im).find("_2") != string::npos) nbin_i=4;
	       }else if((*igm)->getName().find("Disk") != string::npos){
	         nbin_i=((cnt-1)%12)*7; nbin_f=84;
	       }else if((*igm)->getName().find("HalfCylinder") != string::npos){
	         nbin_f=168;
	         if((*im).find("_2") != string::npos) nbin_i=84;
	       }else if((*igm)->getName().find("PixelEndcap") != string::npos){
	         nbin_f=672;
	         if((*im).find("_mO") != string::npos) nbin_i=168;
	         if((*im).find("_pI") != string::npos) nbin_i=336;
	         if((*im).find("_pO") != string::npos) nbin_i=504;
	       }
	       for (int k = 1; k < nbin_f+1; k++) {
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
void SiPixelActionExecutor::getGrandSummaryME(MonitorUserInterface* mui,
    int nbin, string& me_name, vector<MonitorElement*> & mes) {
  vector<string> contents = mui->getMEs();    
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(me_name) == 0) {
      string fullpathname = mui->pwd() + "/" + me_name;
      MonitorElement* me = mui->get(fullpathname);
      if (me) {
	MonitorElementT<TNamed>* obh1 = dynamic_cast<MonitorElementT<TNamed>*> (me);
	if (obh1) {
	  TH1F * root_obh1 = dynamic_cast<TH1F *> (obh1->operator->());
	  if (root_obh1) root_obh1->Reset();        
	  mes.push_back(me);
          return;
	}
      }
    }
  }
  DaqMonitorBEInterface * bei = mui->getBEInterface();
  MonitorElement* temp_me = bei->book1D(me_name.c_str(),me_name.c_str(),nbin,1.,nbin+1.);
  if (temp_me) mes.push_back(temp_me);
}


//
// -- Get Summary ME
//
MonitorElement* SiPixelActionExecutor::getSummaryME(MonitorUserInterface* mui,string me_name) {
//cout<<"Entering SiPixelActionExecutor::getSummaryME..."<<endl;
  MonitorElement* me = 0;
  // If already booked
  vector<string> contents = mui->getMEs();    
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(me_name) == 0) {
      string fullpathname = mui->pwd() + "/" + (*it); 
      me = mui->get(fullpathname);
      if (me) {
	MonitorElementT<TNamed>* obh1 = dynamic_cast<MonitorElementT<TNamed>*> (me);
	if (obh1) {
	  TH1F * root_obh1 = dynamic_cast<TH1F *> (obh1->operator->());
	  if (root_obh1) root_obh1->Reset();        
	}
	return me;
      }
    }
  }
  DaqMonitorBEInterface * bei = mui->getBEInterface();
  me = bei->book1D(me_name.c_str(), me_name.c_str(),4,1.,5.);
  return me;
  //cout<<"...leaving SiPixelActionExecutor::getSummaryME!"<<endl;
}
//
// -- Draw Monitor Elements
//
void SiPixelActionExecutor::drawMEs(int idet, 
  vector<MonitorElement*>& mon_elements, vector<pair <int, float> > & values) {
//cout<<"Entering SiPixelActionExecutor::drawMEs..."<<endl;
  TCanvas canvas("display");
  canvas.Clear();
  if (mon_elements.size() == 2) canvas.Divide(1,2);
  if (mon_elements.size() == 3) canvas.Divide(1,3);
  if (mon_elements.size() == 4) canvas.Divide(2,2);
  if (mon_elements.size() == 5) canvas.Divide(2,3);
  int status;
  int icol;
  string tag;
  
  for (unsigned int i = 0; i < mon_elements.size(); i++) {
    // Mean Value
    float mean_val = mon_elements[i]->getMean();
    // Status after comparison to Referece 
    if (mon_elements[i]->hasError()) {
      status = dqm::qstatus::ERROR;
      tag = "Error";
      icol = 2;
    } else if (mon_elements[i]->hasWarning()) {
      status = dqm::qstatus::WARNING;
      tag = "Warning";
      icol = 5;
    } else  {
      status = dqm::qstatus::STATUS_OK;
      tag = "Ok";
      icol = 3;
    }
//    cout<<"qstatus:"<<status<<" "<<tag<<" "<<icol<<endl;
    // Access the Root object and plot
    MonitorElementT<TNamed>* ob = 
        dynamic_cast<MonitorElementT<TNamed>*>(mon_elements[i]);
    if (ob) {
      canvas.cd(i+1);
      TText tt;
      tt.SetTextSize(0.15);
      tt.SetTextColor(icol);
      ob->operator->()->Draw();
      tt.DrawTextNDC(0.6, 0.5, tag.c_str());
      canvas.Update();
    }
    values.push_back(pair<int,float>(status, mean_val));
  }
  ostringstream name_str;
  name_str << idet << ".jpg";
  canvas.SaveAs(name_str.str().c_str());    
//cout<<"...leaving SiPixelActionExecutor::drawMEs!"<<endl;
  
}

//
// -- Setup Quality Tests 
//
void SiPixelActionExecutor::setupQTests(MonitorUserInterface * mui) {
  mui->cd();
  if (collationDone) mui->cd("Collector/Collated/SiPixel");
  string localPath = string("DQM/SiPixelMonitorClient/test/sipixel_qualitytest_config.xml");
  if(!qtHandler_){
    qtHandler_ = new QTestHandle();
  }
  if(!qtHandler_->configureTests(edm::FileInPath(localPath).fullPath(),mui)){
    cout << " Setting up quality tests " << endl;
    qtHandler_->attachTests(mui);
    mui->cd();
  }else{
    cout << " Problem setting up quality tests "<<endl;
  }
}
//
// -- Check Status of Quality Tests
//
void SiPixelActionExecutor::checkQTestResults(MonitorUserInterface * mui) {
  string currDir = mui->pwd();
  vector<string> contentVec;
  mui->getContents(contentVec);
  for (vector<string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {
    vector<string> contents;
    int nval = SiPixelUtility::getMEList((*it), contents);
    if (nval == 0) continue;
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
      MonitorElement * me = mui->get((*im));
      if (me) {
	// get all warnings associated with me
	vector<QReport*> warnings = me->getQWarnings();
	for(vector<QReport *>::const_iterator it = warnings.begin();
	    it != warnings.end(); ++it) {
	  edm::LogWarning("SiPixelQualityTester::checkTestResults") << 
	    " *** Warning for " << me->getName() << 
	    "," << (*it)->getMessage() << "\n";
	  
	  cout <<  " *** Warning for " << me->getName() << "," 
	       << (*it)->getMessage() << " " << me->getMean() 
	       << " " << me->getRMS() << me->hasWarning() 
	       << endl;
	}
	// get all errors associated with me
	vector<QReport *> errors = me->getQErrors();
	for(vector<QReport *>::const_iterator it = errors.begin();
	    it != errors.end(); ++it) {
	  edm::LogError("SiPixelQualityTester::checkTestResults") << 
	    " *** Error for " << me->getName() << 
	    "," << (*it)->getMessage() << "\n";
	  
	  cout  <<   " *** Error for " << me->getName() << ","
		<< (*it)->getMessage() << " " << me->getMean() 
		<< " " << me->getRMS() 
		<< endl;
	}
      }
    }
  }
}
//
//
//
void SiPixelActionExecutor::createCollation(MonitorUserInterface * mui){
  string currDir = mui->pwd();
  map<string, vector<string> > collation_map;
  vector<string> contentVec;
  mui->getContents(contentVec);

  for (vector<string>::iterator it = contentVec.begin();
      it != contentVec.end(); it++) {
    if ((*it).find("Module_") == string::npos) continue;
    string dir_path;
    vector<string> contents;
//    int nval = SiPixelUtility::getMEList((*it), dir_path, contents);
    string tag = dir_path.substr(dir_path.find("Module_")+7, dir_path.size()-1);
    for (vector<string>::iterator ic = contents.begin(); ic != contents.end(); ic++) {
      
      string me_path = dir_path + (*ic);
      string path = dir_path.substr(dir_path.find("SiPixel"),dir_path.size());
      MonitorElement* me = mui->get( me_path );
      TProfile* prof = ExtractTObject<TProfile>().extract( me );
      TH1F* hist1 = ExtractTObject<TH1F>().extract( me );
      TH2F* hist2 = ExtractTObject<TH2F>().extract( me );
      CollateMonitorElement* coll_me = 0;
      string coll_dir = "Collector/Collated/"+path;
      map<string, vector<string> >::iterator ipos = collation_map.find(tag);
      if(ipos == collation_map.end()) {
        if (collation_map[tag].capacity() != contents.size()) { 
          collation_map[tag].reserve(contents.size()); 
        }
        if      (hist1) coll_me = mui->collate1D((*ic),(*ic),coll_dir);
        else if (hist2) coll_me = mui->collate2D((*ic),(*ic),coll_dir);
        else if (prof) coll_me = mui->collate2D((*ic),(*ic),coll_dir);
        collation_map[tag].push_back(coll_dir+(*ic));
      } else {
        if (find(ipos->second.begin(), ipos->second.end(), (*ic)) == ipos->second.end()){
	  if (hist1)      coll_me = mui->collate1D((*ic),(*ic),coll_dir);
	  else if (hist2) coll_me = mui->collate2D((*ic),(*ic),coll_dir);
	  else if (prof)  coll_me = mui->collateProf((*ic),(*ic),coll_dir);
	  collation_map[tag].push_back(coll_dir+(*ic));	  
        }
      }
      if (coll_me) mui->add(coll_me, me_path);
    }
  }
  collationDone = true;
}
void SiPixelActionExecutor::createLayout(MonitorUserInterface * mui){
  if (configWriter_ == 0) {
    configWriter_ = new SiPixelConfigWriter();
    if (!configWriter_->init()) return;
  }
  string currDir = mui->pwd();   
  if (currDir.find("Layer") != string::npos) {
    string name = "Default";
   configWriter_->createLayout(name);
   configWriter_->createRow();
    fillLayout(mui);
  } else {
    vector<string> subdirs = mui->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      createLayout(mui);
      mui->goUp();
    }
  }  
}
void SiPixelActionExecutor::fillLayout(MonitorUserInterface * mui){
  
  static int icount = 0;
  string currDir = mui->pwd();
  if (currDir.find("Ladder_") != string::npos) {
    vector<string> contents = mui->getMEs(); 
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
    vector<string> subdirs = mui->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      fillLayout(mui);
      mui->goUp();
    }
  }
}
//
// -- Save Monitor Elements in a file
//      
void SiPixelActionExecutor::saveMEs(MonitorUserInterface* mui, string fname){
  if (collationDone) {
    mui->save(fname,"Collector/Collated");
  } else {
     mui->save(fname,mui->pwd(),90);
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

