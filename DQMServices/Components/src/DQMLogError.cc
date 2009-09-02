

#include "DQMServices/Components/src/DQMLogError.h"
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
#include <cmath>

using namespace std;
using namespace edm;



DQMLogError::DQMLogError(const ParameterSet& parameters) {

  // the services
  theDbe = NULL;
  
  categories = NULL;
  modules = NULL;
  total_errors = NULL;  

  //Get from cfg file
  categories_vector = parameters.getParameter< vector<string> >("Categories");
  directoryName = parameters.getParameter<string>("Directory");
}

DQMLogError::~DQMLogError() { 
  // Should the pointers be deleted?
}


void DQMLogError::beginJob(EventSetup const& iSetup) {
   
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
  LogTrace(metname)<<"[DQMLogError] Parameters initialization";
  theDbe = Service<DQMStore>().operator->();
  if(theDbe!=NULL){
    // WHAT SHOULD THIS BE SET TO?
    theDbe->setCurrentFolder(directoryName.append("/LogError")); 

    if(moduleMap.size()!=0){
      modules = theDbe->book1D("modules", "Errors per module", moduleMap.size(), 0, moduleMap.size());    
      for(map<string,int>::const_iterator it = moduleMap.begin(); it!=moduleMap.end();++it){ 
	modules->setBinLabel((*it).second,(*it).first);
      }
    }
    
    if(categoryMap.size()!=0){
      categories = theDbe->book1D("categories", "Errors per category", categoryMap.size(), 0, categoryMap.size());
      for(map<string,int>::const_iterator it = categoryMap.begin(); it!=categoryMap.end();++it){
	categories->setBinLabel((*it).second,(*it).first);
      }
    }

    // HOW MANY BINS SHOULD THE ERROR HIST HAVE?
    int nbins = 10;
    total_errors = theDbe->book1D("total_errors", "Total errors per event", nbins, 0, nbins);
    for(int i=0; i<nbins; ++i){
      stringstream out;
      out<< i;
      string s = out.str();
      total_errors->setBinLabel(i+1,s);
    }
  }
}


void DQMLogError::analyze(const Event& iEvent, const EventSetup& iSetup) {

  LogTrace(metname)<<"[DQMLogError] Analysis of event # ";

  
  // Take the ErrorSummaryEntry container
  Handle<std::vector<edm::ErrorSummaryEntry> >  errors;
  iEvent.getByLabel("logErrorHarvester",errors);
  // Check that errors is valid
  if(!errors.isValid()) return;
  

  // Find the total number of errors in iEvent
  if(errors->size()==0){
    if(total_errors!=NULL){
      total_errors->Fill(0);
    }
  }else{
    int t = 0;
    for (int i=0, n=errors->size(); i<n; i++){
      t+= (*errors)[i].count;
    }
    if(total_errors!=NULL){
      total_errors->Fill(t);
    }    
  }
  
  for(int i=0, n=errors->size(); i< n ; i++){    
    if(errors->size()>0){
      if(categories!=NULL){
	map<string,int>::const_iterator it = categoryMap.find((*errors)[i].category);
	if (it!=categoryMap.end()){
	  // FILL THE RIGHT BIN
	  categories->Fill((*it).second - 1, (*errors)[i].count);
	}
      }
      
      if(modules!=NULL){
	// remove the first part of the module string, what is before ":"
	string s = (*errors)[i].module;
	size_t pos = s.find(':');
	string s_temp = s.substr(pos+1,s.size());
	map<string,int>::const_iterator it = moduleMap.find(s_temp);
	if(it!=moduleMap.end()){
	  // FILL THE RIGHT BIN
	  modules->Fill((*it).second - 1, (*errors)[i].count);
	}
      }
    }
  }

  
  
  
}


void DQMLogError::endJob(void) {
  LogTrace(metname)<<"[DQMLogError] EndJob";
  
}




