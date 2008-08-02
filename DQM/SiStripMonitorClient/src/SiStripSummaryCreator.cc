#include "DQM/SiStripMonitorClient/interface/SiStripSummaryCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

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
  summaryFrequency_ = -1;
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
// -- Read Configuration
//
bool SiStripSummaryCreator::readConfiguration() {
    summaryMEMap.clear();
  SiStripConfigParser config_parser;
  string localPath = string("DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml");
  config_parser.getDocument(edm::FileInPath(localPath).fullPath());
  if (!config_parser.getFrequencyForSummary(summaryFrequency_)){
    cout << "SiStripSummaryCreator::readConfiguration: Failed to read Summary configuration parameters!! ";
    summaryFrequency_ = -1;
    return false;
  }  
  if (!config_parser.getMENamesForSummary(summaryMEMap)) {
    cout << "SiStripSummaryCreator::readConfiguration: Failed to read Summary configuration parameters!! ";
    return false;
  }
  return true;
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
void SiStripSummaryCreator::createSummary(DQMStore* dqm_store) {
  if (summaryMEMap.size() == 0) return;
  string currDir = dqm_store->pwd();
  vector<string> subdirs = dqm_store->getSubdirs();
  int nmod = 0;
  for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
    if ( (*it).find("module_") == string::npos) continue;
    nmod++;       
  }  
  if (nmod > 0) {
    fillSummaryHistos(dqm_store);
  } else {  
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      dqm_store->cd(*it);
      createSummary(dqm_store);
      dqm_store->goUp();
    }
    fillGrandSummaryHistos(dqm_store);
  }
}
//
// -- Create and Fill Summary Histograms at the lowest level of the structure
//
void SiStripSummaryCreator::fillSummaryHistos(DQMStore* dqm_store) {
  string currDir = dqm_store->pwd();
  map<string, MonitorElement*> MEMap;
  vector<string> subdirs = dqm_store->getSubdirs();
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
      dqm_store->cd(*it);
      ndet++;
      vector<MonitorElement*> contents = dqm_store->getContents(dqm_store->pwd());
      dqm_store->goUp();
      for (vector<MonitorElement *>::const_iterator im = contents.begin();
                im != contents.end(); im++) {
        MonitorElement * me_i = (*im);
        if (!me_i) continue;
        string name_i = me_i->getName();
        if (name_i.find(name) == string::npos) continue;
	map<string, MonitorElement*>::iterator iPos = MEMap.find(name); 
	MonitorElement* me;
	// Get the Summary ME
	if (iPos == MEMap.end()){
            me = getSummaryME(dqm_store, name, htype);
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
void SiStripSummaryCreator::fillGrandSummaryHistos(DQMStore* dqm_store) {
  map<string, MonitorElement*> MEMap;
  string currDir = dqm_store->pwd();
  string dir_name =  currDir.substr(currDir.find_last_of("/")+1);
  if ((dir_name.find("SiStrip") == 0) ||
      (dir_name.find("Collector") == 0) ||
      (dir_name.find("MechanicalView") == 0) ||
      (dir_name.find("FU") == 0) ) return;
  vector<string> subdirs = dqm_store->getSubdirs();
  if (subdirs.size() == 0) return;;
  for (map<string,string>::const_iterator isum = summaryMEMap.begin();
       isum != summaryMEMap.end(); isum++) {
    string name, summary_name;
    name = isum->first;
    if (isum->second == "sum" || isum->second == "sum")    
      summary_name = "Summary_" + isum->first;
    else 
      summary_name = "Summary_Mean" + isum->first;
    string htype = isum->second;
    int ibinStep =0;
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      dqm_store->cd(*it);
      vector<MonitorElement*> contents = dqm_store->getContents(dqm_store->pwd());
      dqm_store->goUp();
      for (vector<MonitorElement *>::const_iterator im = contents.begin();
                im != contents.end(); im++) {
        MonitorElement * me_i = (*im);
        if (!me_i) continue;
        string name_i = me_i->getName();
        if (name_i.find((summary_name)) != string::npos) {
          
          map<string, MonitorElement*>::iterator iPos = MEMap.find(name); 
          MonitorElement* me; 
          if (iPos == MEMap.end()) {
            if (htype == "sum" || htype == "Sum") {
	      me = getSummaryME(dqm_store, name, htype);
	    } else {
	      me = getSummaryME(dqm_store, name, "bin-by-bin");              
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
MonitorElement* SiStripSummaryCreator::getSummaryME(DQMStore* dqm_store, 
                         string& name, string htype) {
  MonitorElement* me = 0;
  string currDir = dqm_store->pwd();
  string sum_name, tag_name;
 
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  if (dname.find("_") != string::npos) dname.insert(dname.find("_"),"_");
  if (htype == "sum" && htype == "Sum") {
    sum_name = "Summary" + name + "__" + dname;
    tag_name = "Summary" + name;
  } else {
    sum_name = "Summary_Mean" + name + "__" + dname;
    tag_name = "Summary_Mean" + name;
  }
  // If already booked
  vector<MonitorElement*> contents = dqm_store->getContents(currDir);
  for (vector<MonitorElement *>::const_iterator im = contents.begin();
                im != contents.end(); im++) {
    MonitorElement * me = (*im);
    if (!me)  continue;
    string me_name = me->getName();
    if (me_name.find(sum_name) == 0) {
      if (me->kind() == MonitorElement::DQM_KIND_TH1F ||     
	  me->kind() == MonitorElement::DQM_KIND_TH2F ||
	  me->kind() == MonitorElement::DQM_KIND_TPROFILE) {
	TH1* hist1 = me->getTH1();
	if (hist1) {
	  hist1->Reset();
	  return me;
	}
      }
    }
  }
  map<int, string> tags;
  if (!me) {
    int nBins = 0;
    vector<string> subdirs = dqm_store->getSubdirs();
    // set # of bins of the histogram
    if (htype == "mean" || htype == "Mean" ) {
       nBins = subdirs.size();
       me = dqm_store->book1D(sum_name,sum_name,nBins,0.5,nBins+0.5);
       int ibin = 0;
       for (vector<string>::const_iterator it = subdirs.begin();
          it != subdirs.end(); it++) {
	 string subdir_name = (*it).substr((*it).find_last_of("/")+1);
	 ibin++;
	 tags.insert(pair<int,string>(ibin, (subdir_name)));        
       }
    } else if (htype == "bin-by-bin" || htype == "Bin-by-Bin") {
      for (vector<string>::const_iterator it = subdirs.begin();
	   it != subdirs.end(); it++) {
	dqm_store->cd(*it);
        string subdir_name = (*it).substr((*it).find_last_of("/")+1);
	vector<MonitorElement*> s_contents = dqm_store->getContents(dqm_store->pwd());
	for (vector<MonitorElement *>::const_iterator iv = s_contents.begin();
                iv != s_contents.end(); iv++) {
          MonitorElement* s_me = (*iv);
          if (!s_me) continue;
          string s_me_name = s_me->getName();
	  if (s_me_name.find(name) == 0 || s_me_name.find(tag_name) == 0) {
	    int ibin = s_me->getNbinsX();
	    nBins += ibin;
	    tags.insert(pair<int,string>(nBins-ibin/2, (subdir_name)));        
	    break;
          }
	}
	dqm_store->goUp();
      }
      me = dqm_store->book1D(sum_name,sum_name,nBins,0.5,nBins+0.5);
    } else if (htype == "sum" || htype == "Sum") {
      for (vector<string>::const_iterator it = subdirs.begin();
	   it != subdirs.end(); it++) {
	dqm_store->cd(*it);
	vector<MonitorElement*> s_contents = dqm_store->getContents(dqm_store->pwd());
	dqm_store->goUp();        
	for (vector<MonitorElement *>::const_iterator iv = s_contents.begin();
                iv != s_contents.end(); iv++) {
          MonitorElement* s_me = (*iv);
          if (!s_me) continue;
          string s_me_name = s_me->getName();
          if (s_me_name.find(name) == string::npos) continue;
	  if (s_me->kind() == MonitorElement::DQM_KIND_TH1F) {
            TH1F* hist1 = s_me->getTH1F();
            if (hist1) {
	      nBins = s_me->getNbinsX();
	      me = dqm_store->book1D(sum_name,sum_name,nBins,
		 hist1->GetXaxis()->GetXmin(),hist1->GetXaxis()->GetXmax());
              break;
            }
	  }
        }
      }
    }
  }
  // Set the axis title 
  if (me && me->kind() == MonitorElement::DQM_KIND_TH1F 
      && (htype != "sum" || htype != "Sum")) {
    TH1F* hist = me->getTH1F();
    if (hist) {
      if (name.find("NoisyStrips") != string::npos) hist->GetYaxis()->SetTitle("Noisy Strips (%)");
      else hist->GetYaxis()->SetTitle(name.c_str());
    }
    for (map<int,string>::const_iterator ic = tags.begin();
         ic != tags.end(); ic++) {
      hist->GetXaxis()->SetBinLabel(ic->first, (ic->second).c_str());
    }
    hist->LabelsOption("uv");
  }
  return me;
}
//
// -- Create Layout
//
void SiStripSummaryCreator::createLayout(DQMStore * dqm_store){
  /*  if (configWriter_ == 0) {
    configWriter_ = new SiStripConfigWriter();
    if (!configWriter_->init()) return;
  }
  string currDir = dqm_store->pwd();   
  if (currDir.find("layer") != string::npos) {
    string name = "Default";
   configWriter_->createLayout(name);
   configWriter_->createRow();
    fillLayout(dqm_store);
  } else {
    vector<string> subdirs = dqm_store->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      dqm_store->cd(*it);
      createLayout(dqm_store);
      dqm_store->goUp();
    }
  } 
  string fname = "test.xml";
  configWriter_->write(fname); 
  if (configWriter_) delete configWriter_;
  configWriter_ = 0;*/
}
//
//
//
void SiStripSummaryCreator::fillHistos(int ival, int istep, string htype, 
                       MonitorElement* me_src, MonitorElement* me) {
  
  if (me->getTH1()) {
    TProfile* prof = 0;
    TH1F* hist1 = 0;
    TH2F* hist2 = 0;
    if (me->kind() == MonitorElement::DQM_KIND_TH1F)    hist1 = me->getTH1F();
    if (me->kind() == MonitorElement::DQM_KIND_TH2F)    hist2 = me->getTH2F();
    if (me->kind() == MonitorElement::DQM_KIND_TPROFILE) prof = me->getTProfile();
    
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
}
