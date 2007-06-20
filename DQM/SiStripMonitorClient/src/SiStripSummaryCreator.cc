#include "DQM/SiStripMonitorClient/interface/SiStripSummaryCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"

#include <iostream>
using namespace std;
//
// -- Constructor
// 
SiStripSummaryCreator::SiStripSummaryCreator() {
  edm::LogInfo("SiStripSummaryCreator") << 
    " Creating SiStripSummaryCreator " << "\n" ;
  summaryMEMap.clear();
  configWriter_ = 0;
}
//
// --  Destructor
// 
SiStripSummaryCreator::~SiStripSummaryCreator() {
  edm::LogInfo("SiStripSummaryCreator") << 
    " Deleting SiStripSummaryCreator " << "\n" ;
  summaryMEMap.clear();
  if (configWriter_) delete configWriter_;
}
//
// -- Set Summary ME names
//
void SiStripSummaryCreator::setSummaryMENames(map<string, string>& me_names) {

  summaryMEMap.clear();
  for (map<string,string>::const_iterator isum = me_names.begin();
       isum != me_names.end(); isum++) {    
    summaryMEMap.insert(pair<string,string>(isum->first, isum->second));
  }
}
//
// -- Browse through the Folder Structure
//
void SiStripSummaryCreator::createSummary(MonitorUserInterface* mui) {
  string currDir = mui->pwd();
  vector<string> subdirs = mui->getSubdirs();
  int nmod = 0;
  for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
    if ( (*it).find("module_") == string::npos) continue;
    nmod++;       
  }  
  if (nmod > 0) {
    fillSummaryHistos(mui);
  } else {  
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      mui->cd(*it);
      createSummary(mui);
      mui->goUp();
    }
    fillGrandSummaryHistos(mui);
  }
}
//
// -- Create and Fill Summary Histograms at the lowest level of the structure
//
void SiStripSummaryCreator::fillSummaryHistos(MonitorUserInterface* mui) {
  string currDir = mui->pwd();
  map<string, MonitorElement*> MEMap;
  vector<string> subdirs = mui->getSubdirs();
  if (subdirs.size() ==0) return;
  

  for (map<string,string>::const_iterator isum = summaryMEMap.begin();
       isum != summaryMEMap.end(); isum++) {    
    string name = isum->first;
    int iBinStep = 0;
    int ndet = 0;
    string htype = isum->second;
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      if ( (*it).find("module_") == string::npos) continue;
      mui->cd(*it);
      ndet++;
      vector<string> contents = mui->getMEs();    
      mui->goUp();
      for (vector<string>::const_iterator im = contents.begin();
	   im != contents.end(); im++) {
        if ((*im).find(name) == string::npos) continue;
	string full_path = mui->pwd() + "/" +(*it)+ "/" + (*im);
	MonitorElement * me_i = mui->get(full_path);
	if (!me_i) continue;
	map<string, MonitorElement*>::iterator iPos = MEMap.find(name); 
	MonitorElement* me;
	// Get the Summary ME
	if (iPos == MEMap.end()){
            me = getSummaryME(mui, name, htype);
            MEMap.insert(pair<string, MonitorElement*>(name, me));
	} else  me =  iPos->second;
	// Fill it now
        fillHistos(ndet, iBinStep, htype, me_i, me); 
	iBinStep += me_i->getNbinsX();
	break;
      }
    }
  }
}
//
//  -- Fill Summary Histogram at higher level
//
void SiStripSummaryCreator::fillGrandSummaryHistos(MonitorUserInterface* mui) {
  map<string, MonitorElement*> MEMap;
  string currDir = mui->pwd();
  string dir_name =  currDir.substr(currDir.find_last_of("/")+1);
  if ((dir_name.find("SiStrip") == 0) ||
      (dir_name.find("Collector") == 0) ||
      (dir_name.find("MechanicalView") == 0) ||
      (dir_name.find("FU") == 0) ) return;
  vector<string> subdirs = mui->getSubdirs();
  if (subdirs.size() == 0) return;;
  for (map<string,string>::const_iterator isum = summaryMEMap.begin();
       isum != summaryMEMap.end(); isum++) {    
    string name = isum->first;
    string htype = isum->second;
    int ibinStep =0;
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      vector<string> contents = mui->getMEs();
      mui->goUp();
      for (vector<string>::const_iterator im = contents.begin();
	   im != contents.end(); im++) {
	if ((*im).find((name)) != string::npos) {
	  string full_path = currDir + "/" + (*it) + "/" +(*im);
	  MonitorElement * me_i = mui->get(full_path);
	  if (!me_i) continue;
          
          map<string, MonitorElement*>::iterator iPos = MEMap.find(name); 
          MonitorElement* me; 
          if (iPos == MEMap.end()) {
            if (htype == "sum" || htype == "Sum") {
	      me = getSummaryME(mui, name, htype);
	    } else {
	      me = getSummaryME(mui, name, "bin-by-bin");              
            }
	    MEMap.insert(pair<string, MonitorElement*>(name, me));
          } else  me =  iPos->second;
          if (htype == "sum" || htype == "Sum") {
	    fillHistos(0, ibinStep, htype, me_i, me);
	  } else {
	    fillHistos(0, ibinStep,"bin-by-bin", me_i, me);
          }
          ibinStep += me_i->getNbinsX();
          break;
	}
      }
    }
  }
}
//
// -- Get Summary ME
//
MonitorElement* SiStripSummaryCreator::getSummaryME(MonitorUserInterface* mui, 
                         string& name, string htype) {
  MonitorElement* me = 0;
  string currDir = mui->pwd();
  string sum_name = "Summary_" + name + "_in_" 
                      + currDir.substr(currDir.find_last_of("/")+1);
  // If already booked
  vector<string> contents = mui->getMEs();    
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(sum_name) == 0) {
      string fullpath = currDir + "/" + (*it); 
      me = mui->get(fullpath);
      if (me) {	
	TH1F* hist1 = ExtractTObject<TH1F>().extract(me);
	if (hist1) {
          hist1->Reset();
          hist1->LabelsOption("uv");
          return me;
	}
      }
    }
  }
  if (!me) {
    DaqMonitorBEInterface * bei = mui->getBEInterface();
    int nBins = 0;
    vector<string> subdirs = mui->getSubdirs();
    map<int, string> tags;
    
    // set # of bins of the histogram
    if (htype == "mean" || htype == "Mean" ) {
       nBins = subdirs.size();
       me = bei->book1D(sum_name,sum_name,nBins,0.5,nBins+0.5);
    } else if (htype == "bin-by-bin" || htype == "Bin-by-Bin") {
      for (vector<string>::const_iterator it = subdirs.begin();
	   it != subdirs.end(); it++) {
	mui->cd(*it);
	vector<string> s_contents = mui->getMEs();    
	for (vector<string>::const_iterator iv = s_contents.begin();
	     iv != s_contents.end(); iv++) {
	  if ((*iv).find(name) == string::npos) continue;
	  
	  string sub_path =   mui->pwd() + "/" + (*iv);
	  MonitorElement* s_me = mui->get(sub_path);
	  if (s_me) {
	    int ibin = s_me->getNbinsX();
	    nBins += ibin;
	    tags.insert(pair<int,string>(nBins-ibin/2, (*it)));        
	    break;
	  }
	}
	mui->goUp();
      }
      me = bei->book1D(sum_name,sum_name,nBins,0.5,nBins+0.5);
    } else if (htype == "sum" || htype == "Sum") {
      for (vector<string>::const_iterator it = subdirs.begin();
	   it != subdirs.end(); it++) {
	mui->cd(*it);
	vector<string> s_contents = mui->getMEs();    
	for (vector<string>::const_iterator iv = s_contents.begin();
	     iv != s_contents.end(); iv++) {
	  if ((*iv).find(name) == string::npos) continue;
	  
	  string sub_path =   mui->pwd() + "/" + (*iv);
	  MonitorElement* s_me = mui->get(sub_path);
	  if (s_me) {
            TH1F* hist1 = ExtractTObject<TH1F>().extract( s_me );
            if (hist1) {
	      nBins = s_me->getNbinsX();
	      me = bei->book1D(sum_name,sum_name,nBins,
		 hist1->GetXaxis()->GetXmin(),hist1->GetXaxis()->GetXmax());
              break;
            }
	  }
        }
      }
    }
    // Set the axis title 
    if (me) { 
      TH1F* hist = ExtractTObject<TH1F>().extract( me );
      if (hist) hist->GetYaxis()->SetTitle(name.c_str());
    }   
    for (map<int,string>::const_iterator ic = tags.begin();
      ic != tags.end(); ic++) {
      me->setBinLabel(ic->first, ic->second);
    }
  }
  return me;
}
//
// -- Create Layout
//
void SiStripSummaryCreator::createLayout(MonitorUserInterface * mui){
  if (configWriter_ == 0) {
    configWriter_ = new SiStripConfigWriter();
    if (!configWriter_->init()) return;
  }
  string currDir = mui->pwd();   
  if (currDir.find("layer") != string::npos) {
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
  string fname = "test.xml";
  configWriter_->write(fname); 
  if (configWriter_) delete configWriter_;
  configWriter_ = 0;
}
//
// -- Fill Layout
//
void SiStripSummaryCreator::fillLayout(MonitorUserInterface * mui){
  
  static int icount = 0;
  string currDir = mui->pwd();
  if (currDir.find("string_") != string::npos) {
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
//
//
void SiStripSummaryCreator::fillHistos(int ival, int istep, string htype, 
                       MonitorElement* me_src, MonitorElement* me) {
  
  TProfile* prof = ExtractTObject<TProfile>().extract( me_src );
  TH1F* hist1 = ExtractTObject<TH1F>().extract( me_src );
  TH2F* hist2 = ExtractTObject<TH2F>().extract( me_src );   
  int nbins = me_src->getNbinsX();
  string name = me_src->getName();
  if (htype == "mean" || htype == "Mean" ) {
    if (hist2 &&  name.find("NoisyStrips") != string::npos) {
      float bad = 0.0;
      float entries = me_src->getEntries();
      if (entries > 0.0) {
	float binEntry = entries/nbins;
	for (int k=1; k<nbins+1; k++) {
	  float noisy = me_src->getBinContent(k,3)+me_src->getBinContent(k,5);
	  float dead = me_src->getBinContent(k,2)+me_src->getBinContent(k,4);
	  float good = me_src->getBinContent(k,1);
	  if (noisy >= binEntry*0.5 || dead >= binEntry*0.5) bad++;
	}
	bad = bad*100.0/nbins;    
	me->Fill(ival, bad);
      }
    } else me->Fill(ival, me_src->getMean());
  } else if (htype == "bin-by-bin" || htype == "Bin-by-Bin") {
    for (int k=1; k<nbins+1; k++) {
      me->setBinContent(istep+k,me_src->getBinContent(k));
    }
  } else if (htype == "sum" || htype == "Sum") {  
    if ( hist1) {
      for (int k=1; k<nbins+1; k++) {
        float val = me_src->getBinContent(k) + me->getBinContent(k) ;
        me->setBinContent(k,val);
      }
    }        
  }
}
