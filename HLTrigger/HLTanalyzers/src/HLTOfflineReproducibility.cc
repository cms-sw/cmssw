// -*- C++ -*-
//
// Package:    HLTOfflineReproducibility
// Class:      HLTOfflineReproducibility
// 
/**\class HLTOfflineReproducibility HLTOfflineReproducibility.cc HLTOfflineReproducibility/src/HLTOfflineReproducibility.cc

 Description: compares online and offline HLT trigger results

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Juliette Marie Alimena,40 3-A02,+41227671577,
//         Created:  Fri Apr 22 15:46:58 CEST 2011
// $Id: HLTOfflineReproducibility.cc,v 1.5 2011/11/15 11:18:38 fwyzard Exp $
//
//

using namespace std;

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/HLTGlobalStatus.h"
#include "DataFormats/HLTReco/interface/HLTPrescaleTable.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/HLTConfigData.h"

#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TH2.h>
#include <vector>
#include <iostream>
#include <ostream>
//
// class declaration
//

class HLTOfflineReproducibility : public edm::EDAnalyzer {
public:
  explicit HLTOfflineReproducibility(const edm::ParameterSet&);
  ~HLTOfflineReproducibility();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  
  bool check(std::string process, std::string pCheck);

  // ----------member data ---------------------------  
  edm::InputTag triggerLabelON_;
  edm::InputTag triggerLabelOFF_;
  
  //Trigger Stuff
  unsigned int nPaths_, nPathsON_, nPathsOFF_, nDatasets_;
  vector<string> triggerNames_;
  vector< vector<string> > moduleLabel_, moduleLabelON_, moduleLabelOFF_;
  vector<unsigned int> nModules_, nPaths_PD_;
  vector<string> datasetNames_;
  vector<vector<string> > datasetContent_;
  vector< vector< vector<bool> > > triggerNames_matched_;

  string processNameON_;
  string processNameOFF_;
  HLTConfigProvider hltConfig_;

  int Nfiles_;
  double Normalization_;
  bool isRealData_;
  int LumiSecNumber_; 
  vector<int> trigger_online_;
  vector<int> trigger_offline_;

  TH1D* path_ON_hist;
  TH1D* path_ONnotOFF_hist;
  TH1D* path_OFFnotON_hist;
  TH2D* pathmodule_ONnotOFF_hist;
  TH2D* pathmodule_OFFnotON_hist;

  vector<TH1D*> path_ON_hist_PD;
  vector<TH1D*> path_ONnotOFF_hist_PD;
  vector<TH1D*> path_OFFnotON_hist_PD;
  vector<TH2D*> pathmodule_ONnotOFF_hist_PD;
  vector<TH2D*> pathmodule_OFFnotON_hist_PD;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HLTOfflineReproducibility::HLTOfflineReproducibility(const edm::ParameterSet& iConfig):  
  triggerLabelON_       (iConfig.getUntrackedParameter<edm::InputTag>("triggerTagON")),
  triggerLabelOFF_      (iConfig.getUntrackedParameter<edm::InputTag>("triggerTagOFF")), 
  nPaths_               (0),
  nDatasets_            (0),
  triggerNames_         (),
  moduleLabel_          (),
  nModules_             (), 
  nPaths_PD_            (),
  datasetNames_         (),
  datasetContent_       (),
  triggerNames_matched_ (),
  processNameON_(iConfig.getParameter<std::string>("processNameON")),
  processNameOFF_       (iConfig.getParameter<std::string>("processNameOFF")),
  Nfiles_               (iConfig.getUntrackedParameter<int>("Nfiles",0)),
  Normalization_        (iConfig.getUntrackedParameter<double>("Norm",40.)),
  isRealData_           (iConfig.getUntrackedParameter<bool>("isRealData",true)),
  LumiSecNumber_        (iConfig.getUntrackedParameter<int>("LumiSecNumber",1)),
  path_ON_hist          (0),
  path_ONnotOFF_hist    (0),
  path_OFFnotON_hist    (0),
  pathmodule_ONnotOFF_hist(0),
  pathmodule_OFFnotON_hist(0),
  path_ON_hist_PD       (),
  path_ONnotOFF_hist_PD (),
  path_OFFnotON_hist_PD (),
  pathmodule_ONnotOFF_hist_PD(),
  pathmodule_OFFnotON_hist_PD()
{
}


HLTOfflineReproducibility::~HLTOfflineReproducibility()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
HLTOfflineReproducibility::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  //cout <<"Run/Event/Lumi block "<< iEvent.id().run()<<" "<<iEvent.id().event()<<" "<<iEvent.luminosityBlock() <<endl;
  
  //Initialize Trigger
  TriggerResults trON_;
  Handle<TriggerResults> h_trigResON_;
  iEvent.getByLabel(triggerLabelON_, h_trigResON_);
  trON_ = *h_trigResON_;  
  
  TriggerResults trOFF_;
  Handle<TriggerResults> h_trigResOFF_;
  iEvent.getByLabel(triggerLabelOFF_, h_trigResOFF_);  
  trOFF_ = *h_trigResOFF_;

  vector<string> triggerListON_;
  Service<service::TriggerNamesService> tnsON_;
  bool foundNamesON_ = tnsON_->getTrigPaths(trON_,triggerListON_);
  if (!foundNamesON_) cout << "Could not get trigger names!\n";
  if (trON_.size()!=triggerListON_.size()) cout << "ERROR: length of names and paths not the same: " << triggerListON_.size() << "," << trON_.size() << endl;  
  
  vector<string> triggerListOFF_;
  Service<service::TriggerNamesService> tnsOFF_;
  bool foundNamesOFF_ = tnsOFF_->getTrigPaths(trOFF_,triggerListOFF_);
  if (!foundNamesOFF_) cout << "Could not get trigger names!\n";
  if (trOFF_.size()!=triggerListOFF_.size()) cout << "ERROR: length of names and paths not the same: " << triggerListOFF_.size() << "," << trOFF_.size() << endl;  
  

  vector<bool> online_accept_, offline_accept_, fails_prescaler_; 
  vector<int> module_indexON_, module_indexOFF_;
  vector<string> module_index_labelON_, module_index_labelOFF_;
  
  for (unsigned int x=0; x<nPaths_; x++) {
    online_accept_.push_back(false);
    offline_accept_.push_back(false);
    module_indexON_.push_back(-1);
    module_indexOFF_.push_back(-1);
    module_index_labelON_.push_back(" ");
    module_index_labelOFF_.push_back(" ");
    fails_prescaler_.push_back(false);
  }
  
  //loop over online trigger paths
  for (unsigned int i=0; i<nPathsON_; i++) {
    for (unsigned int x=0; x<nPaths_; x++) {
      if (triggerListON_[i]==triggerNames_[x]) {

	//if online accepted
	if (trON_[i].wasrun()==1 && trON_[i].accept()==1 && trON_[i].error()==0) {
	  online_accept_.at(x) = true;
	  trigger_online_[x]++;
	  path_ON_hist->Fill(x);
	  for (unsigned int a=0; a<nDatasets_; a++) {
	    for (unsigned int b=0; b<nPaths_PD_[a]; b++) {
	      if (triggerNames_matched_[i][a][b]) path_ON_hist_PD[a]->Fill(b);
	    }
	  }
	}
	
	//if online failed
	if (trON_[i].accept() == 0){
	  module_index_labelON_.at(x) = moduleLabelON_[i][trON_[i].index()];
	  module_indexON_.at(x) = hltConfig_.moduleIndex(triggerNames_[x],module_index_labelON_[x]);
	}
	
	//for each path, loop over modules and find if a path fails on a prescaler
	for (unsigned int j=0; j<nModules_[x]; ++j) {
	  //const string& moduleLabel_(moduleLabel[j]);
	  const string& moduleType = hltConfig_.moduleType(moduleLabel_[x][j]);
	  if ( (trON_[i].accept()==0 && j==trON_[i].index()) && (moduleType=="HLTPrescaler" || moduleType=="TriggerResultsFilter") ) fails_prescaler_[x] = true;
	}

      }
    }
  }


  //loop over offline trigger paths
  for (unsigned int i=0; i<nPathsOFF_; i++) {
    for (unsigned int x=0; x<nPaths_; x++) {
      if (triggerListOFF_[i]==triggerNames_[x]) {

	//if offline accepted
	if (trOFF_[i].wasrun()==1 && trOFF_[i].accept()==1 && trOFF_[i].error()==0) {
	  offline_accept_[x] = true;
	  trigger_offline_[x]++;
	}

	//if offline failed
	if (trOFF_[i].accept() == 0) {
	  module_index_labelOFF_.at(x) = moduleLabelOFF_[i][trOFF_[i].index()];
	  module_indexOFF_.at(x) = hltConfig_.moduleIndex(triggerNames_[x],module_index_labelOFF_[x]);
	}
	
	//for each path, loop over modules and find if a path fails on a prescaler
	for (unsigned int j=0; j<nModules_[x]; ++j) {
	  //const string& moduleLabel_(moduleLabel[j]);
	  const string& moduleType = hltConfig_.moduleType(moduleLabel_[x][j]);
	  if ( (trOFF_[i].accept()==0 && j==trOFF_[i].index()) && (moduleType=="HLTPrescaler" || moduleType=="TriggerResultsFilter") ) fails_prescaler_[x] = true;
	}

      }
    }
  }



  //check agreement between online and offline
  //loop over trigger paths (online and offline)
  for (unsigned int x=0; x<nPaths_; x++) {
    if (!fails_prescaler_[x]){ //ignore paths that fail on a prescale
      if(online_accept_[x] && !offline_accept_[x]){ //online fires but offline doesn't
	path_ONnotOFF_hist->Fill(x);
	pathmodule_ONnotOFF_hist->Fill(x,module_indexOFF_[x]); //module and path for where it fails offline
	cout<<"  Event "<<iEvent.id().event()<<" in run "<<iEvent.id().run()<<" and luminosity block "<<iEvent.luminosityBlock()<<endl;
	cout<<"  fires online but not offline!!"<<endl;
	cout<<"  Path is: "<<triggerNames_[x]<<", last run module is: "<<module_index_labelOFF_[x]<<endl;
	for (unsigned int a=0; a<nDatasets_; a++) {
	  for (unsigned int b=0; b<nPaths_PD_[a]; b++) {
	    if (triggerNames_matched_[x][a][b]){
	      path_ONnotOFF_hist_PD[a]->Fill(b);
	      pathmodule_ONnotOFF_hist_PD[a]->Fill(b,module_indexOFF_[x]); //module and path for where it fails offline
	    }
	  }
	}
      }
      if(!online_accept_[x] && offline_accept_[x]){//offline fires but online doesn't
	path_OFFnotON_hist->Fill(x);
	pathmodule_OFFnotON_hist->Fill(x,module_indexON_[x]); //module and path for where it fails online
	cout<<"  Event "<<iEvent.id().event()<<" in run "<<iEvent.id().run()<<" and luminosity block "<<iEvent.luminosityBlock()<<endl;
	cout<<"  fires offline but not online!!"<<endl;
	cout<<"  Path is: "<<triggerNames_[x]<<", last run module is: "<<module_index_labelON_[x]<<endl;
	for (unsigned int a=0; a<nDatasets_; a++) {
	  for (unsigned int b=0; b<nPaths_PD_[a]; b++) {
	    if (triggerNames_matched_[x][a][b]){
	      path_OFFnotON_hist_PD[a]->Fill(b);
	      pathmodule_OFFnotON_hist_PD[a]->Fill(b,module_indexON_[x]); //module and path for where it fails online
	    }
	  }
	}
      }
    }

  }//end of loop over trigger paths



  //const vector<string> & moduleLabels(hltConfig_.moduleLabels(i));
  //cout<<"triggerListON_["<<i<<"] is: "<<triggerListON_[i]<<endl;
  

}


// ------------ method called once each job just before starting event loop  ------------
void 
HLTOfflineReproducibility::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTOfflineReproducibility::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
HLTOfflineReproducibility::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  edm::Service<TFileService> fs;  
  void init(const edm::TriggerResults &, const edm::TriggerNames & HLTNames);

  bool changed(true);
  nPathsON_=0, nPathsOFF_=0;
  vector<string> triggerNamesON_, triggerNamesOFF_;
  vector<string> temp_;
  unsigned int max_nModules_=0;
  vector<unsigned int> max_nModules_PD_, nModules_diff_;
  bool TriggerModuleNamesOK_ = true;

  //---------------------------------------hltConfig for online-------------------------------
  if (hltConfig_.init(iRun,iSetup,processNameON_,changed)) {
    // if init returns TRUE, initialisation has succeeded!
    if (changed) {
      cout<<"hlt_Config_.init returns true for online"<<endl;
      // The HLT config has actually changed wrt the previous Run, hence rebook your
      // histograms or do anything else dependent on the revised HLT config
      // check if trigger name in (new) config
      triggerNamesON_ = hltConfig_.triggerNames();

      //loop over trigger paths
      nPathsON_ = hltConfig_.size();
      for (unsigned int i=0; i<nPathsON_; i++) {
	temp_.clear();
	//const vector<string> & moduleLabelsON(hltConfig_.moduleLabels(i));
	for (unsigned int j=0; j<hltConfig_.moduleLabels(i).size(); j++) {
	  temp_.push_back(hltConfig_.moduleLabel(i,j));
	}
	moduleLabelON_.push_back(temp_);
      }

    }
  } 
  else {
    // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
    // with the file and/or code and needs to be investigated!
    cout<<" HLT config extraction failure with process name " << processNameON_;
    // In this case, all access methods will return empty values!
  }

  //-------------------hltConfig for offline----------------------------------------
  if (hltConfig_.init(iRun,iSetup,processNameOFF_,changed)) {
    // if init returns TRUE, initialisation has succeeded!
    if (changed) {
      cout<<"hlt_Config_.init returns true for offline"<<endl;
      // The HLT config has actually changed wrt the previous Run, hence rebook your
      // histograms or do anything else dependent on the revised HLT config
      // check if trigger name in (new) config
      triggerNamesOFF_ = hltConfig_.triggerNames();

      //loop over trigger paths
      nPathsOFF_ = hltConfig_.size();
      for (unsigned int i=0; i<nPathsOFF_; i++) {
	temp_.clear();
	for (unsigned int j=0; j<hltConfig_.moduleLabels(i).size(); j++){
	  temp_.push_back(hltConfig_.moduleLabel(i,j));
	}
	moduleLabelOFF_.push_back(temp_);
      }

    }
  } 
  else {
    // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
    // with the file and/or code and needs to be investigated!
    cout<<" HLT config extraction failure with process name " << processNameOFF_;
    // In this case, all access methods will return empty values!
  }

  //------------------compare online and offline hltConfig------------------------
  if (nPathsON_==0 || nPathsOFF_==0){
    cout<<"There are 0 paths online or offline!! There are "<<nPathsON_<<" paths online and "<<nPathsOFF_<<" paths offline!!!"<<endl;
    TriggerModuleNamesOK_ = false;
  }
  else{
    //define nPaths_ as number of paths shared between online and offline
    if (nPathsON_<=nPathsOFF_) nPaths_=nPathsON_;
    else nPaths_=nPathsOFF_;

    for (unsigned int i=0; i<nPathsON_; i++) {
      for (unsigned int j=0; j<nPathsOFF_; j++) {
	if (triggerNamesON_[i]==triggerNamesOFF_[j]){
	  triggerNames_.push_back(triggerNamesON_[i]);
	  if (i!=j) cout<<"Path "<<triggerNamesON_[i]<<" corresponds to path number "<<i<<" for online and path number "<<j<<" for offline"<<endl;

	  //define nModules_ as number of modules shared between online and offline
	  if (moduleLabelON_[i].size()<=moduleLabelOFF_[j].size()) nModules_.push_back(moduleLabelON_[i].size());
	  else nModules_.push_back(moduleLabelOFF_[j].size());

	  if (nModules_[i]>max_nModules_) max_nModules_=nModules_[i];
	  
	  if (moduleLabelON_[i].size()>moduleLabelOFF_[j].size()) nModules_diff_.push_back(moduleLabelON_[i].size()-moduleLabelOFF_[j].size());
	  else nModules_diff_.push_back(moduleLabelOFF_[j].size()-moduleLabelON_[i].size());

	  temp_.clear();
	  for (unsigned int a=0; a<moduleLabelON_[i].size(); a++) {
	    for (unsigned int b=0; b<moduleLabelOFF_[j].size(); b++) {
	      //match online and offline module labels
	      //since a module can be run twice per path, but not usually right after one another,
	      //require that a and b be fairly close to each other, +/- the difference in the number of modules run offline vs online,
	      //to avoid double-counting modules that are repeated later in the path
	      //also, since we need to work with unsigned ints, a or b could also be 0, ignoring the requirement described above
	      if ( (moduleLabelON_[i][a]==moduleLabelOFF_[j][b]) && ( (b<=a+nModules_diff_[i] && b>=a-nModules_diff_[i]) || (a==0 || b==0) ) ){
		temp_.push_back(moduleLabelON_[i][a]);
		if (a!=b){
		  cout<<"For path "<<triggerNamesON_[i]<<" online and "<<triggerNamesOFF_[j]<<" offline:"<<endl;
		  cout<<"  module "<<moduleLabelON_[i][a]<<" corresponds to module number "<<a<<" for online and module number "<<b<<" for offline"<<endl;
		}
	      }
	    }
	  }
	  moduleLabel_.push_back(temp_);
	}
      }
    }
  }


  //----------------------------------------------------------------------------------------------------------

  //if everything is good:
  //save path and module names to be used in event loop, print out the module names and types for each path
  //get paths in each dataset
  if (TriggerModuleNamesOK_){

    //------------all paths--------------
    cout<<endl<<"There are "<<nPaths_<<" paths in total"<<endl;
    cout<<"Maximum number of modules over all paths is: "<<max_nModules_<<endl;  

    for (unsigned int x=0; x<nPaths_; x++) {
      trigger_online_.push_back(0);
      trigger_offline_.push_back(0);
      cout<<endl<<"For "<<triggerNames_[x]<<" (trigger number "<<x<<"), there are "<<nModules_[x]<<" modules:"<<endl;
      for (unsigned int j=0; j<nModules_[x]; j++) {
	const string& moduleType_ = hltConfig_.moduleType(moduleLabel_[x][j]);
	cout<<"  module "<<j<<" is "<<moduleLabel_[x][j]<<" and is of type "<<moduleType_<<endl;
      }
    }


    //---------paths per dataset-------------------------------------------
    //get datasets, initialize max_nModules_PD_
    datasetNames_ = hltConfig_.datasetNames();
    nDatasets_ = datasetNames_.size();    
    for (unsigned int a=0; a<nDatasets_; a++) {
      max_nModules_PD_.push_back(0);
    }

    //loop over datasets, paths in each dataset
    for (unsigned int a=0; a<nDatasets_; a++) {
      temp_.clear();
      vector<string> datasetcontent_ = hltConfig_.datasetContent(a);
      nPaths_PD_.push_back(datasetcontent_.size());
      cout<<endl<<"For dataset "<<datasetNames_[a]<<" (dataset number "<<a<<"), there are "<<nPaths_PD_[a]<<" paths:"<<endl;
      for (unsigned int b=0; b<nPaths_PD_[a]; b++) {
	cout<<"  path "<<b<<" is "<<datasetcontent_[b]<<endl;
	temp_.push_back(datasetcontent_[b]);
      }
      datasetContent_.push_back(temp_);
    }

    //match trigger names in full list to trigger names per dataset; find max number of modules over all triggers in each dataset
    //find matches
    vector <vector<bool> > temp1_;
    vector<bool> temp2_;
    for (unsigned int x=0; x<nPaths_; x++) {
      temp1_.clear();
      for (unsigned int a=0; a<nDatasets_; a++) {
	temp2_.clear();
	for (unsigned int b=0; b<nPaths_PD_[a]; b++) {
	  if (triggerNames_[x]==datasetContent_[a][b]){
	    temp2_.push_back(true);	    
	    //cout<<"Matched trigger name is: "<<datasetContent_[a][b]<<" for dataset "<<a<<" and dataset path "<<b<<endl;
	  }
	  else temp2_.push_back(false);
	}
	temp1_.push_back(temp2_);
      }
      triggerNames_matched_.push_back(temp1_);
    }

    //if matched and # of modules is bigger than all previous ones, take that number as new maximum
    for (unsigned int x=0; x<nPaths_; x++) {
      for (unsigned int a=0; a<nDatasets_; a++) {
	for (unsigned int b=0; b<nPaths_PD_[a]; b++) {
	  if (triggerNames_matched_[x][a][b] && nModules_[x]>max_nModules_PD_[a]) max_nModules_PD_[a]=nModules_[x];
	}
      }
    }

    //for (unsigned int a=0; a<nDatasets_; a++) {
    //cout<<"For dataset "<<datasetNames_[a]<<", the max number of modules is: "<<max_nModules_PD_[a]<<endl;
    //}

  }//end if all triggers and modules match from online to offline
  

  //----------------------------------------------------------------------------------------------------------


  //define histograms

  //all paths
  path_ON_hist = fs->make<TH1D>("path_ON_hist", "Total Times Online Path Fires", nPaths_, 0, nPaths_);
  path_ONnotOFF_hist = fs->make<TH1D>("path_ONnotOFF_hist", "Online Path fires but Offline does not", nPaths_, 0, nPaths_);
  path_OFFnotON_hist = fs->make<TH1D>("path_OFFnotON_hist", "Offline Path fires but Online does not", nPaths_, 0, nPaths_);
  pathmodule_ONnotOFF_hist = fs->make<TH2D>("pathmodule_ONnotOFF_hist", "Last run module index vs Path for Offline, when Online fired but Offline didn't", nPaths_, 0, nPaths_, max_nModules_, 0, max_nModules_);
  pathmodule_OFFnotON_hist = fs->make<TH2D>("pathmodule_OFFnotON_hist", "Last run module index vs Path for Online, when Offline fired but Online didn't", nPaths_, 0, nPaths_, max_nModules_, 0, max_nModules_);
  
  //paths per dataset
  char path_ON_name[100], path_ONnotOFF_name[100], path_OFFnotON_name[100], pathmodule_ONnotOFF_name[100], pathmodule_OFFnotON_name[100];
  for (unsigned int a = 0; a < nDatasets_; ++a) {
    snprintf(path_ON_name,             100, "path_ON_hist_PD[%i]",              a);
    snprintf(path_ONnotOFF_name,       100, "path_ONnotOFF_hist_PD[%i]",        a);
    snprintf(path_OFFnotON_name,       100, "path_OFFnotON_hist_PD[%i]",        a);
    snprintf(pathmodule_ONnotOFF_name, 100, "pathmodule_ONnotOFF_hist_PD[%i]",  a);
    snprintf(pathmodule_OFFnotON_name, 100, "pathmodule_OFFnotON_hist_PD[%i]",  a);

    TString path_ON_title             = "Total Times Online Path Fires (" + datasetNames_[a] + " dataset)";
    TString path_ONnotOFF_title       = "Online Path fires but Offline does not (" + datasetNames_[a] + " dataset)";
    TString path_OFFnotON_title       = "Offline Path fires but Online does not (" + datasetNames_[a] + " dataset)";
    TString pathmodule_ONnotOFF_title = "Last run module index vs Path for Offline, when Online fired but Offline didn't (" + datasetNames_[a] + " dataset)";
    TString pathmodule_OFFnotON_title = "Last run module index vs Path for Online, when Offline fired but Online didn't (" + datasetNames_[a] + " dataset)";

    path_ON_hist_PD.push_back(fs->make<TH1D>(path_ON_name, path_ON_title, nPaths_PD_[a], 0, nPaths_PD_[a]));
    path_ONnotOFF_hist_PD.push_back(fs->make<TH1D>(path_ONnotOFF_name, path_ONnotOFF_title, nPaths_PD_[a], 0, nPaths_PD_[a]));
    path_OFFnotON_hist_PD.push_back(fs->make<TH1D>(path_OFFnotON_name, path_OFFnotON_title, nPaths_PD_[a], 0, nPaths_PD_[a]));
    pathmodule_ONnotOFF_hist_PD.push_back(fs->make<TH2D>(pathmodule_ONnotOFF_name, pathmodule_ONnotOFF_title, nPaths_PD_[a], 0, nPaths_PD_[a], max_nModules_PD_[a], 0, max_nModules_PD_[a]));
    pathmodule_OFFnotON_hist_PD.push_back(fs->make<TH2D>(pathmodule_OFFnotON_name, pathmodule_OFFnotON_title, nPaths_PD_[a], 0, nPaths_PD_[a], max_nModules_PD_[a], 0, max_nModules_PD_[a]));
  }
  
}

// ------------ method called when ending the processing of a run  ------------
void 
HLTOfflineReproducibility::endRun(edm::Run const&, edm::EventSetup const&)
{
  //all paths
  cout<<endl;
  for (unsigned int x=0; x<nPaths_; x++) {
    cout<<triggerNames_[x]<<" online accepts: "<<trigger_online_[x]<<", offline accepts: "<<trigger_offline_[x]<<endl;
    path_ON_hist->GetXaxis()->SetBinLabel(x+1,triggerNames_[x].c_str());
    path_ONnotOFF_hist->GetXaxis()->SetBinLabel(x+1,triggerNames_[x].c_str());
    path_OFFnotON_hist->GetXaxis()->SetBinLabel(x+1,triggerNames_[x].c_str());
    pathmodule_ONnotOFF_hist->GetXaxis()->SetBinLabel(x+1,triggerNames_[x].c_str());
    pathmodule_OFFnotON_hist->GetXaxis()->SetBinLabel(x+1,triggerNames_[x].c_str());
  }

  //paths per dataset
  for (unsigned int a=0; a<nDatasets_; a++) {
    for (unsigned int b=0; b<nPaths_PD_[a]; b++) {
      path_ON_hist_PD[a]->GetXaxis()->SetBinLabel(b+1,datasetContent_[a][b].c_str());
      path_ONnotOFF_hist_PD[a]->GetXaxis()->SetBinLabel(b+1,datasetContent_[a][b].c_str());
      path_OFFnotON_hist_PD[a]->GetXaxis()->SetBinLabel(b+1,datasetContent_[a][b].c_str());
      pathmodule_ONnotOFF_hist_PD[a]->GetXaxis()->SetBinLabel(b+1,datasetContent_[a][b].c_str());
      pathmodule_OFFnotON_hist_PD[a]->GetXaxis()->SetBinLabel(b+1,datasetContent_[a][b].c_str());
    }
  }

}

// ------------ method called when starting to processes a luminosity block  ------------
void 
HLTOfflineReproducibility::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
HLTOfflineReproducibility::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HLTOfflineReproducibility::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTOfflineReproducibility);
