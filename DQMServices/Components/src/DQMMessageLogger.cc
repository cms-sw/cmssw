

#include "DQMServices/Components/src/DQMMessageLogger.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "TPad.h"
#include <cmath>

using namespace std;
using namespace edm;



DQMMessageLogger::DQMMessageLogger(const ParameterSet& parameters) {

  // the services
  theDbe = NULL;
  
  categories_errors = NULL;
  categories_warnings = NULL;
  modules_errors = NULL;
  modules_warnings = NULL;
  total_errors = NULL;  
  total_warnings = NULL;

  //Get from cfg file
  categories_vector = parameters.getParameter< vector<string> >("Categories");
  directoryName = parameters.getParameter<string>("Directory");
  
}

DQMMessageLogger::~DQMMessageLogger() { 
  // Should the pointers be deleted?
}


void DQMMessageLogger::beginJob() {
   
  metname = "errorAnalyzer";



  // MAKE CATEGORYMAP USING INPUT FROM CFG FILE
  for(unsigned int i=0; i<categories_vector.size(); i++){
    categoryMap.insert(pair<string,int>(categories_vector[i],i+1));
  }
  
 
  // MAKE MODULEMAP
  typedef Service<edm::service::TriggerNamesService>  TNS;
  typedef vector<std::string> stringvec;
  TNS tns;
  stringvec const& trigpaths = tns->getTrigPaths();
  
  
  for (stringvec::const_iterator i = trigpaths.begin(), e =trigpaths.end() ;  i != e;  ++i){
      stringvec strings =  tns->getTrigPathModules(*i);

      for(unsigned int k=0; k<strings.size(); ++k){      
	moduleMap.insert(pair<string,int>(strings[k],moduleMap.size()+1));
      }    
  }

  // BOOK THE HISTOGRAMS
  LogTrace(metname)<<"[DQMMessageLogger] Parameters initialization";
  theDbe = Service<DQMStore>().operator->();
  if(theDbe!=NULL){
    
    

    if(moduleMap.size()!=0){
      theDbe->setCurrentFolder(directoryName + "/Errors"); 
      modules_errors = theDbe->book1D("modules_errors", "Errors per module", moduleMap.size(), 0, moduleMap.size()); 
      theDbe->setCurrentFolder(directoryName + "/Warnings"); 
      
      modules_warnings = theDbe->book1D("modules_warnings","Warnings per module",moduleMap.size(),0,moduleMap.size());
      
      for(map<string,int>::const_iterator it = moduleMap.begin(); it!=moduleMap.end();++it){ 
	modules_errors->setBinLabel((*it).second,(*it).first);
        modules_warnings->setBinLabel((*it).second,(*it).first);
      }
      modules_errors->getTH1()->GetXaxis()->LabelsOption("v");
      modules_warnings->getTH1()->GetXaxis()->LabelsOption("v");
      
      
      
      
    }
    
    if(categoryMap.size()!=0){
      theDbe->setCurrentFolder(directoryName + "/Errors"); 
      categories_errors = theDbe->book1D("categories_errors", "Errors per category", categoryMap.size(), 0, categoryMap.size());
      theDbe->setCurrentFolder(directoryName +"/Warnings"); 
      categories_warnings = theDbe->book1D("categories_warnings", "Warnings per category", categoryMap.size(), 0, categoryMap.size());
      
      
      for(map<string,int>::const_iterator it = categoryMap.begin(); it!=categoryMap.end();++it){
	categories_errors->setBinLabel((*it).second,(*it).first);
	categories_warnings->setBinLabel((*it).second,(*it).first);
      }
      categories_warnings->getTH1()->GetXaxis()->LabelsOption("v");
      categories_errors->getTH1()->GetXaxis()->LabelsOption("v");
    }

    // HOW MANY BINS SHOULD THE ERROR HIST HAVE?
    int nbins = 11;
    total_warnings = theDbe->book1D("total_warnings","Total warnings per event",nbins,-0.5,nbins+0.5);
    theDbe->setCurrentFolder(directoryName + "/Errors"); 
    total_errors = theDbe->book1D("total_errors", "Total errors per event", nbins, -0.5, nbins+0.5);
    
    for(int i=0; i<nbins; ++i){
      stringstream out;
      out<< i;
      string s = out.str();
      total_errors->setBinLabel(i+1,s);
      total_warnings->setBinLabel(i+1,s);
    }
  }
}


void DQMMessageLogger::analyze(const Event& iEvent, const EventSetup& iSetup) {

  LogTrace(metname)<<"[DQMMessageLogger] Analysis of event # ";

  
  // Take the ErrorSummaryEntry container
  Handle<std::vector<edm::ErrorSummaryEntry> >  errors;
  iEvent.getByLabel("logErrorHarvester",errors);
  // Check that errors is valid
  if(!errors.isValid()){   return; }
  // Compare severity level of error with ELseveritylevel instance el : "-e" should be the lowest error
  ELseverityLevel el("-e");

  

  // Find the total number of errors in iEvent
  if(errors->size()==0){
    if(total_errors!=NULL){
      total_errors->Fill(0);
    }
    if(total_warnings!=NULL){
      total_warnings->Fill(0);
    }
  }else{
    
    int e = 0;
    int w = 0;
    for (int i=0, n=errors->size(); i<n; i++){
      if((*errors)[i].severity.getLevel() < el.getLevel()){
	w+= (*errors)[i].count;
      }else{
	e+= (*errors)[i].count;
      }
    }
    if(total_errors!=NULL){
      total_errors->Fill(e);
    }    
    if(total_warnings!=NULL){
      total_warnings->Fill(w);
    }
  }

  
  
  
  for(int i=0, n=errors->size(); i< n ; i++){    
    
    //cout << "Severity for error/warning: " << (*errors)[i].severity << " " <<(*errors)[i].module  << endl;

    if(errors->size()>0){
      // IF THIS IS AN ERROR on the ELseverityLevel SCALE, FILL ERROR HISTS
      if((*errors)[i].severity.getLevel() >= el.getLevel()){
	if(categories_errors!=NULL){
	  map<string,int>::const_iterator it = categoryMap.find((*errors)[i].category);
	  if (it!=categoryMap.end()){
	    // FILL THE RIGHT BIN
	    categories_errors->Fill((*it).second - 1, (*errors)[i].count);
	  }
	}
	//	if (categoryECount.size()<=40)
	//	  categoryECount[(*errors)[i].category]+=(*errors)[i].count;

	if(modules_errors!=NULL){
	  // remove the first part of the module string, what is before ":"
	  string s = (*errors)[i].module;
	  size_t pos = s.find(':');
	  string s_temp = s.substr(pos+1,s.size());
	  map<string,int>::const_iterator it = moduleMap.find(s_temp);
	  if(it!=moduleMap.end()){
	    // FILL THE RIGHT BIN
	    modules_errors->Fill((*it).second - 1, (*errors)[i].count);
	  }
	}
	// IF ONLY WARNING, FILL WARNING HISTS
      }else{
	if(categories_warnings!=NULL){
	  map<string,int>::const_iterator it = categoryMap.find((*errors)[i].category);
	  if (it!=categoryMap.end()){
	    // FILL THE RIGHT BIN
	    categories_warnings->Fill((*it).second - 1, (*errors)[i].count);
	  }
	}

	//	if (categoryWCount.size()<=40)
	//	  categoryWCount[(*errors)[i].category]+=(*errors)[i].count;
	
	if(modules_warnings!=NULL){
	  // remove the first part of the module string, what is before ":"
	  string s = (*errors)[i].module;
	  size_t pos = s.find(':');
	  string s_temp = s.substr(pos+1,s.size());
	  map<string,int>::const_iterator it = moduleMap.find(s_temp);
	  if(it!=moduleMap.end()){
	    // FILL THE RIGHT BIN
	    modules_warnings->Fill((*it).second - 1, (*errors)[i].count);
	  }
	}
      }
    }
  }
}

void DQMMessageLogger::endRun(const edm::Run & , const edm::EventSetup & ){
  /*
  theDbe = Service<DQMStore>().operator->();
  if(theDbe!=NULL){
    std::map<std::string,int>::iterator it;
    uint i=0;
    theDbe->setCurrentFolder(directoryName + "/Errors");
    if (categoryECount.empty()){
      MonitorElement * catECount = theDbe->book1D("categoryCount_errors","Errors per Category",1,0,1);
      catECount->setBinLabel(1,"No Errors");
    }else{
      MonitorElement * catECount = theDbe->book1D("categoryCount_errors","Errors per Category",categoryECount.size(),0,categoryECount.size());
      for (i=1,it=categoryECount.begin();it!=categoryECount.end();++it,++i){
	catECount->setBinLabel(i,it->first);
	catECount->setBinContent(i,it->second);
      }
    }
    theDbe->setCurrentFolder(directoryName + "/Warnings");
    if (categoryWCount.empty()){
      MonitorElement * catWCount = theDbe->book1D("categoryCount_warnings","Warnings per Category",categoryWCount.size(),0,categoryWCount.size());
      catWCount->setBinLabel(1,"No Warnings");
    }else{
      MonitorElement * catWCount = theDbe->book1D("categoryCount_warnings","Warnings per Category",categoryWCount.size(),0,categoryWCount.size());
      for (i=1,it=categoryWCount.begin();it!=categoryWCount.end();++it,++i){
	catWCount->setBinLabel(i,it->first);
	catWCount->setBinContent(i,it->second);
      }
    }
  }
  categoryWCount.clear();
  categoryECount.clear();
  */
}

void DQMMessageLogger::endJob(void) {
  LogTrace(metname)<<"[DQMMessageLogger] EndJob";
  
}




