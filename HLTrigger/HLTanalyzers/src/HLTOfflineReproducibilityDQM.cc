// -*- C++ -*-
//
// Package:    HLTOfflineReproducibility_DQM
// Class:      HLTOfflineReproducibility_DQM
// 
/**\class HLTOfflineReproducibility_DQM HLTOfflineReproducibility_DQM.cc HLTOfflineReproducibility_DQM/src/HLTOfflineReproducibility_DQM.cc

 Description: compares online and offline HLT trigger results, within the DQM framework

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Juliette Marie Alimena,40 3-A02,+41227671577,
//         Created:  Tues Oct 11 15:46:58 CEST 2011
// $Id: HLTOfflineReproducibilityDQM.cc,v 1.3 2011/11/15 11:33:52 jalimena Exp $
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

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

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

class HLTOfflineReproducibilityDQM : public edm::EDAnalyzer {
public:
  explicit HLTOfflineReproducibilityDQM(const edm::ParameterSet&);
  ~HLTOfflineReproducibilityDQM();
  
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
  DQMStore* dqms_;

  edm::InputTag triggerLabelORIG_;
  edm::InputTag triggerLabelNEW_;
  
  //Trigger Stuff
  unsigned int nPaths_, nPathsORIG_, nPathsNEW_, nDatasets_;
  vector<string> triggerNames_;
  vector< vector<string> > moduleLabel_, moduleLabelORIG_, moduleLabelNEW_;
  vector<unsigned int> nModules_, nPaths_PD_;
  vector<string> datasetNames_;
  vector<vector<string> > datasetContent_;
  vector< vector< vector<bool> > > triggerNames_matched_;

  string processNameORIG_;
  string processNameNEW_;
  edm::InputTag triggerResultsTag_;
  edm::InputTag triggerEventTag_;
  HLTConfigProvider hltConfig_;

  int Nfiles_;
  double Normalization_;
  bool isRealData_;
  int LumiSecNumber_; 
  vector<int> trigger_ORIG_;
  vector<int> trigger_NEW_;

  MonitorElement* path_ORIG_hist;
  MonitorElement* path_ORIGnotNEW_hist;
  MonitorElement* path_NEWnotORIG_hist;
  MonitorElement* pathmodule_ORIGnotNEW_hist;
  MonitorElement* pathmodule_NEWnotORIG_hist;

  vector<MonitorElement*> path_ORIG_hist_PD;
  vector<MonitorElement*> path_ORIGnotNEW_hist_PD;
  vector<MonitorElement*> path_NEWnotORIG_hist_PD;
  vector<MonitorElement*> pathmodule_ORIGnotNEW_hist_PD;
  vector<MonitorElement*> pathmodule_NEWnotORIG_hist_PD;

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
HLTOfflineReproducibilityDQM::HLTOfflineReproducibilityDQM(const edm::ParameterSet& iConfig):  
  triggerLabelORIG_       (iConfig.getUntrackedParameter<edm::InputTag>("triggerTagORIG")),
  triggerLabelNEW_      (iConfig.getUntrackedParameter<edm::InputTag>("triggerTagNEW")), 
  nPaths_               (0),
  nDatasets_            (0),
  triggerNames_         (),
  moduleLabel_          (),
  nModules_             (), 
  nPaths_PD_            (),
  datasetNames_         (),
  datasetContent_       (),
  triggerNames_matched_ (),
  processNameORIG_(iConfig.getParameter<std::string>("processNameORIG")),
  processNameNEW_       (iConfig.getParameter<std::string>("processNameNEW")),
  Nfiles_               (iConfig.getUntrackedParameter<int>("Nfiles",0)),
  Normalization_        (iConfig.getUntrackedParameter<double>("Norm",40.)),
  isRealData_           (iConfig.getUntrackedParameter<bool>("isRealData",true)),
  LumiSecNumber_        (iConfig.getUntrackedParameter<int>("LumiSecNumber",1)),
  path_ORIG_hist          (0),
  path_ORIGnotNEW_hist    (0),
  path_NEWnotORIG_hist    (0),
  pathmodule_ORIGnotNEW_hist(0),
  pathmodule_NEWnotORIG_hist(0),
  path_ORIG_hist_PD       (),
  path_ORIGnotNEW_hist_PD (),
  path_NEWnotORIG_hist_PD (),
  pathmodule_ORIGnotNEW_hist_PD(),
  pathmodule_NEWnotORIG_hist_PD()
{
  //now do what ever initialization is needed
  //define parameters

  dqms_ = edm::Service<DQMStore>().operator->();
}


HLTOfflineReproducibilityDQM::~HLTOfflineReproducibilityDQM()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
HLTOfflineReproducibilityDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  //cout <<"Run/Event/Lumi block "<< iEvent.id().run()<<" "<<iEvent.id().event()<<" "<<iEvent.luminosityBlock() <<endl;
  
  //Initialize Trigger
  TriggerResults trORIG_;
  Handle<TriggerResults> h_trigResORIG_;
  iEvent.getByLabel(triggerLabelORIG_, h_trigResORIG_);
  trORIG_ = *h_trigResORIG_;  
  
  TriggerResults trNEW_;
  Handle<TriggerResults> h_trigResNEW_;
  iEvent.getByLabel(triggerLabelNEW_, h_trigResNEW_);  
  trNEW_ = *h_trigResNEW_;

  vector<string> triggerListORIG_;
  Service<service::TriggerNamesService> tnsORIG_;
  bool foundNamesORIG_ = tnsORIG_->getTrigPaths(trORIG_,triggerListORIG_);
  if (!foundNamesORIG_) cout << "Could not get trigger names!\n";
  if (trORIG_.size()!=triggerListORIG_.size()) cout << "ERROR: length of names and paths not the same: " << triggerListORIG_.size() << "," << trORIG_.size() << endl;  
  
  vector<string> triggerListNEW_;
  Service<service::TriggerNamesService> tnsNEW_;
  bool foundNamesNEW_ = tnsNEW_->getTrigPaths(trNEW_,triggerListNEW_);
  if (!foundNamesNEW_) cout << "Could not get trigger names!\n";
  if (trNEW_.size()!=triggerListNEW_.size()) cout << "ERROR: length of names and paths not the same: " << triggerListNEW_.size() << "," << trNEW_.size() << endl;  
  
  vector<bool> ORIG_accept_, NEW_accept_, fails_prescaler_; 
  vector<int> module_indexORIG_, module_indexNEW_;
  vector<string> module_index_labelORIG_, module_index_labelNEW_;
  
  for (unsigned int x=0; x<nPaths_; x++) {
    ORIG_accept_.push_back(false);
    NEW_accept_.push_back(false);
    module_indexORIG_.push_back(-1);
    module_indexNEW_.push_back(-1);
    module_index_labelORIG_.push_back(" ");
    module_index_labelNEW_.push_back(" ");
    fails_prescaler_.push_back(false);
  }

  //loop over ORIG trigger paths
  for (unsigned int i=0; i<nPathsORIG_; i++) {
    for (unsigned int x=0; x<nPaths_; x++) {
      //match to list of paths that are in common to both ORIG and NEW
      if (triggerListORIG_.at(i)==triggerNames_.at(x)) {

        //if ORIG accepted
        if (trORIG_.at(i).wasrun()==1 && trORIG_.at(i).accept()==1 && trORIG_.at(i).error()==0) {
          ORIG_accept_.at(x) = true;
          trigger_ORIG_.at(x)++;
          path_ORIG_hist->Fill(x);
          for (unsigned int a=0; a<nDatasets_; a++) {
            for (unsigned int b=0; b<nPaths_PD_.at(a); b++) {
              if (triggerNames_matched_.at(i).at(a).at(b)) path_ORIG_hist_PD.at(a)->Fill(b);
            }
          }
        }
        
        //if ORIG failed
        if (trORIG_.at(i).accept() == 0){
          module_index_labelORIG_.at(x) = moduleLabelORIG_.at(i)[trORIG_.at(i).index()];
          module_indexORIG_.at(x) = hltConfig_.moduleIndex(triggerNames_.at(x),module_index_labelORIG_.at(x));
        }

        //for each path, loop over modules and find if a path fails on a prescaler
        for (unsigned int j=0; j<nModules_.at(x); ++j) {
          const string& moduleType = hltConfig_.moduleType(moduleLabel_.at(x).at(j));
          if ( (trORIG_.at(i).accept()==0 && j==trORIG_.at(i).index()) && (moduleType=="HLTPrescaler" || moduleType=="TriggerResultsFilter") ) fails_prescaler_.at(x) = true;
        }

      }
    }
  }


  //loop over NEW trigger paths
  for (unsigned int i=0; i<nPathsNEW_; i++) {
    for (unsigned int x=0; x<nPaths_; x++) {
      //match to list of paths that are in common to both ORIG and NEW
      if (triggerListNEW_.at(i)==triggerNames_.at(x)) {

        //if NEW accepted
        if (trNEW_.at(i).wasrun()==1 && trNEW_.at(i).accept()==1 && trNEW_.at(i).error()==0) {
          NEW_accept_.at(x) = true;
          trigger_NEW_.at(x)++;
        }

        //if NEW failed
        if (trNEW_.at(i).accept() == 0) {
          module_index_labelNEW_.at(x) = moduleLabelNEW_.at(i)[trNEW_.at(i).index()];
          module_indexNEW_.at(x) = hltConfig_.moduleIndex(triggerNames_.at(x),module_index_labelNEW_.at(x));
        }
        
        //for each path, loop over modules and find if a path fails on a prescaler
        for (unsigned int j=0; j<nModules_.at(x); ++j) {
          const string& moduleType = hltConfig_.moduleType(moduleLabel_.at(x).at(j));
          if ( (trNEW_.at(i).accept()==0 && j==trNEW_.at(i).index()) && (moduleType=="HLTPrescaler" || moduleType=="TriggerResultsFilter") ) fails_prescaler_.at(x) = true;
        }

      }
    }
  }


  //check agreement between ORIG and NEW
  //loop over trigger paths common to both ORIG and NEW
  for (unsigned int x=0; x<nPaths_; x++) {
    if (!fails_prescaler_.at(x)){ //ignore paths that fail on a prescale
      if(ORIG_accept_.at(x) && !NEW_accept_.at(x)){ //ORIG fires but NEW doesn't
        path_ORIGnotNEW_hist->Fill(x);
        pathmodule_ORIGnotNEW_hist->Fill(x,module_indexNEW_.at(x)); //module and path for where it fails in NEW
        cout<<"  Event "<<iEvent.id().event()<<" in run "<<iEvent.id().run()<<" and luminosity block "<<iEvent.luminosityBlock()<<endl;
        cout<<"  fires in ORIG but not NEW!!"<<endl;
        cout<<"  Path is: "<<triggerNames_.at(x)<<", last run module is: "<<module_index_labelNEW_.at(x)<<endl;
        for (unsigned int a=0; a<nDatasets_; a++) {
          for (unsigned int b=0; b<nPaths_PD_.at(a); b++) {
            if (triggerNames_matched_.at(x).at(a).at(b)){
              path_ORIGnotNEW_hist_PD.at(a)->Fill(b);
              pathmodule_ORIGnotNEW_hist_PD.at(a)->Fill(b,module_indexNEW_.at(x)); //module and path for where it fails in NEW
            }
          }
        }
      }
      if(!ORIG_accept_.at(x) && NEW_accept_.at(x)){//NEW fires but ORIG doesn't
        path_NEWnotORIG_hist->Fill(x);
        pathmodule_NEWnotORIG_hist->Fill(x,module_indexORIG_.at(x)); //module and path for where it fails ORIG
        cout<<"  Event "<<iEvent.id().event()<<" in run "<<iEvent.id().run()<<" and luminosity block "<<iEvent.luminosityBlock()<<endl;
        cout<<"  fires in NEW but not ORIG!!"<<endl;
        cout<<"  Path is: "<<triggerNames_.at(x)<<", last run module is: "<<module_index_labelORIG_.at(x)<<endl;
        for (unsigned int a=0; a<nDatasets_; a++) {
          for (unsigned int b=0; b<nPaths_PD_.at(a); b++) {
            if (triggerNames_matched_.at(x).at(a).at(b)){
              path_NEWnotORIG_hist_PD.at(a)->Fill(b);
              pathmodule_NEWnotORIG_hist_PD.at(a)->Fill(b,module_indexORIG_.at(x)); //module and path for where it fails ORIG
            }
          }
        }
      }
    }

  }//end of loop over trigger paths



  //const vector<string> & moduleLabels(hltConfig_.moduleLabels(i));
  //const string& moduleLabel_(moduleLabel.at(j));


}


// ------------ method called once each job just before starting event loop  ------------
void 
HLTOfflineReproducibilityDQM::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTOfflineReproducibilityDQM::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
HLTOfflineReproducibilityDQM::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  void init(const edm::TriggerResults &, const edm::TriggerNames & HLTNames);

  bool changed(true);
  nPathsORIG_=0, nPathsNEW_=0;
  vector<string> triggerNamesORIG_, triggerNamesNEW_;
  vector<string> temp_;
  unsigned int max_nModules_=0;
  vector<unsigned int> max_nModules_PD_, nModules_diff_;
  bool TriggerModuleNamesOK_ = true;

  //---------------------------------------hltConfig for ORIG-------------------------------
  if (hltConfig_.init(iRun,iSetup,processNameORIG_,changed)) {
    // if init returns TRUE, initialisation has succeeded!
    if (changed) {
      cout<<"hlt_Config_.init returns true for ORIG"<<endl;
      // The HLT config has actually changed wrt the previous Run, hence rebook your
      // histograms or do anything else dependent on the revised HLT config
      // check if trigger name in (new) config
      triggerNamesORIG_ = hltConfig_.triggerNames();

      //loop over trigger paths
      nPathsORIG_ = hltConfig_.size();
      for (unsigned int i=0; i<nPathsORIG_; i++) {
	temp_.clear();
	//const vector<string> & moduleLabelsORIG(hltConfig_.moduleLabels(i));
	for (unsigned int j=0; j<hltConfig_.moduleLabels(i).size(); j++) {
	  temp_.push_back(hltConfig_.moduleLabel(i,j));
	}
	moduleLabelORIG_.push_back(temp_);
      }

    }
  } 
  else {
    // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
    // with the file and/or code and needs to be investigated!
    cout<<" HLT config extraction failure with process name " << processNameORIG_;
    // In this case, all access methods will return empty values!
  }

  //-------------------hltConfig for NEW----------------------------------------
  if (hltConfig_.init(iRun,iSetup,processNameNEW_,changed)) {
    // if init returns TRUE, initialisation has succeeded!
    if (changed) {
      cout<<"hlt_Config_.init returns true for NEW"<<endl;
      // The HLT config has actually changed wrt the previous Run, hence rebook your
      // histograms or do anything else dependent on the revised HLT config
      // check if trigger name in (new) config
      triggerNamesNEW_ = hltConfig_.triggerNames();

      //loop over trigger paths
      nPathsNEW_ = hltConfig_.size();
      for (unsigned int i=0; i<nPathsNEW_; i++) {
	temp_.clear();
	for (unsigned int j=0; j<hltConfig_.moduleLabels(i).size(); j++){
	  temp_.push_back(hltConfig_.moduleLabel(i,j));
	}
	moduleLabelNEW_.push_back(temp_);
      }

    }
  } 
  else {
    // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
    // with the file and/or code and needs to be investigated!
    cout<<" HLT config extraction failure with process name " << processNameNEW_;
    // In this case, all access methods will return empty values!
  }

  //------------------compare ORIG and NEW hltConfig------------------------
  if (nPathsORIG_==0 || nPathsNEW_==0){
    cout<<"There are 0 paths ORIG or in NEW!! There are "<<nPathsORIG_<<" paths ORIG and "<<nPathsNEW_<<" paths in NEW!!!"<<endl;
    TriggerModuleNamesOK_ = false;
  }
  else{
    //define nPaths_ as number of paths shared between ORIG and NEW
    if (nPathsORIG_<=nPathsNEW_) nPaths_=nPathsORIG_;
    else nPaths_=nPathsNEW_;

    for (unsigned int i=0; i<nPathsORIG_; i++) {
      for (unsigned int j=0; j<nPathsNEW_; j++) {
	//match ORIG and NEW paths
        if (triggerNamesORIG_.at(i)==triggerNamesNEW_.at(j)){
          triggerNames_.push_back(triggerNamesORIG_.at(i));
          if (i!=j) cout<<"Path "<<triggerNamesORIG_.at(i)<<" corresponds to path number "<<i<<" for ORIG and path number "<<j<<" for NEW"<<endl;

          //define nModules_ as number of modules shared between ORIG and in NEW
          if (moduleLabelORIG_.at(i).size()<=moduleLabelNEW_.at(j).size()) nModules_.push_back(moduleLabelORIG_.at(i).size());
          else nModules_.push_back(moduleLabelNEW_.at(j).size());

	  //define max number of modules over all the paths
          if (nModules_.at(i)>max_nModules_) max_nModules_=nModules_.at(i);

	  //define the abs(number_of_modules_run_in_ORIG_for_this_path - number_of_modules_run_in_NEW_for_this_path)
          if (moduleLabelORIG_.at(i).size()>moduleLabelNEW_.at(j).size()) nModules_diff_.push_back(moduleLabelORIG_.at(i).size()-moduleLabelNEW_.at(j).size());
          else nModules_diff_.push_back(moduleLabelNEW_.at(j).size()-moduleLabelORIG_.at(i).size());

          temp_.clear();
          for (unsigned int a=0; a<moduleLabelORIG_.at(i).size(); a++) {
            for (unsigned int b=0; b<moduleLabelNEW_.at(j).size(); b++) {
              //match ORIG and NEW modules
              //since a module can be run twice per path, but not usually right after one another,
              //require that a and b be fairly close to each other, +/- the difference in the number of modules run in ORIG vs NEW,
              //to avoid double-counting modules that are repeated later in the path
              //also, since we need to work with unsigned ints, a or b could also be 0, ignoring the requirement described above
              if (moduleLabelORIG_.at(i).at(a)==moduleLabelNEW_.at(j).at(b) && ( (b<=a+nModules_diff_.at(i) && b>=a-nModules_diff_.at(i)) || (a==0 || b==0) ) ){
                temp_.push_back(moduleLabelORIG_.at(i).at(a));
                if (a!=b){
                  cout<<"For path "<<triggerNamesORIG_.at(i)<<" in ORIG and "<<triggerNamesNEW_.at(j)<<" in NEW:"<<endl;
                  cout<<"  module "<<moduleLabelORIG_.at(i).at(a)<<" corresponds to module number "<<a<<" for ORIG and module number "<<b<<" for NEW"<<endl;
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
    cout<<"There are "<<nPaths_<<" paths in total"<<endl;
    cout<<"Maximum number of modules over all paths is: "<<max_nModules_<<endl;

    for (unsigned int i=0; i<nPaths_; i++) {
      trigger_ORIG_.push_back(0);
      trigger_NEW_.push_back(0);
      cout<<endl<<"For "<<triggerNames_.at(i)<<" (trigger number "<<i<<"), there are "<<nModules_.at(i)<<" modules:"<<endl;
      for (unsigned int j=0; j<nModules_.at(i); j++) {
	const string& moduleType_ = hltConfig_.moduleType(moduleLabel_.at(i).at(j));
	cout<<"  module "<<j<<" is "<<moduleLabel_.at(i).at(j)<<" and is of type "<<moduleType_<<endl;
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
      cout<<endl<<"For dataset "<<datasetNames_.at(a)<<" (dataset number "<<a<<"), there are "<<nPaths_PD_.at(a)<<" paths:"<<endl;
      for (unsigned int b=0; b<nPaths_PD_.at(a); b++) {
	cout<<"  path "<<b<<" is "<<datasetcontent_.at(b)<<endl;
	temp_.push_back(datasetcontent_.at(b));
      }
      datasetContent_.push_back(temp_);
    }

    //match trigger names in full list to trigger names per dataset; find max number of modules over all triggers in each dataset
    //find matches
    vector <vector<bool> > temp1_;
    vector<bool> temp2_;
    for (unsigned int i=0; i<nPaths_; i++) {
      temp1_.clear();
      for (unsigned int a=0; a<nDatasets_; a++) {
	temp2_.clear();
	for (unsigned int b=0; b<nPaths_PD_.at(a); b++) {
	  if (triggerNames_.at(i)==datasetContent_.at(a).at(b)){
	    temp2_.push_back(true);	    
	    //cout<<"Matched trigger name is: "<<datasetContent_.at(a).at(b)<<" for dataset "<<a<<" and dataset path "<<b<<endl;
	  }
	  else temp2_.push_back(false);
	}
	temp1_.push_back(temp2_);
      }
      triggerNames_matched_.push_back(temp1_);
    }

    //if matched and # of modules is bigger than all previous ones, take that number as new maximum
    for (unsigned int i=0; i<nPaths_; i++) {
      for (unsigned int a=0; a<nDatasets_; a++) {
	for (unsigned int b=0; b<nPaths_PD_.at(a); b++) {
	  if (triggerNames_matched_.at(i).at(a).at(b) && nModules_.at(i)>max_nModules_PD_.at(a)) max_nModules_PD_.at(a)=nModules_.at(i);
	}
      }
    }

    //for (unsigned int a=0; a<nDatasets_; a++) {
    //cout<<"For dataset "<<datasetNames_.at(a)<<", the max number of modules is: "<<max_nModules_PD_.at(a)<<endl;
    //}

  }//end if all triggers and modules match from ORIG to NEW


  //----------------------------------------------------------------------------------------------------------


  //define histograms
  if (!dqms_) return;

  //all paths
  dqms_->setCurrentFolder("DQMExample/DQMSource_HLTOfflineReproducibility");

  path_ORIG_hist = dqms_->book1D("path_ORIG_hist","Total Times Path Fires in ORIG",nPaths_,0,nPaths_);
  path_ORIGnotNEW_hist = dqms_->book1D("path_ORIGnotNEW_hist","Path fires in ORIG but not in NEW",nPaths_,0,nPaths_);
  path_NEWnotORIG_hist = dqms_->book1D("path_NEWnotORIG_hist","Path fires in NEW but not in ORIG",nPaths_,0,nPaths_);
  pathmodule_ORIGnotNEW_hist = dqms_->book2D("pathmodule_ORIGnotNEW_hist","Last run module index vs Path for NEW, when ORIG fired but NEW didn't",nPaths_,0,nPaths_,max_nModules_,0,max_nModules_);
  pathmodule_NEWnotORIG_hist = dqms_->book2D("pathmodule_NEWnotORIG_hist","Last run module index vs Path for ORIG, when NEW fired but ORIG didn't",nPaths_,0,nPaths_,max_nModules_,0,max_nModules_);
  
  //paths per dataset
  char folder_name[500];
  char path_ORIG_name[100], path_ORIGnotNEW_name[100], path_NEWnotORIG_name[100], pathmodule_ORIGnotNEW_name[100], pathmodule_NEWnotORIG_name[100];
  for (unsigned int a=0; a<nDatasets_; a++) {
    sprintf(folder_name,"DQMExample/DQMSource_HLTOfflineReproducibility/%s",datasetNames_.at(a).c_str());
    dqms_->setCurrentFolder(folder_name);

    snprintf(path_ORIG_name,             100, "path_ORIG_hist_%s",              datasetNames_.at(a).c_str());
    snprintf(path_ORIGnotNEW_name,       100, "path_ORIGnotNEW_hist_%s",        datasetNames_.at(a).c_str());
    snprintf(path_NEWnotORIG_name,       100, "path_NEWnotORIG_hist_%s",        datasetNames_.at(a).c_str());
    snprintf(pathmodule_ORIGnotNEW_name, 100, "pathmodule_ORIGnotNEW_hist_%s",  datasetNames_.at(a).c_str());
    snprintf(pathmodule_NEWnotORIG_name, 100, "pathmodule_NEWnotORIG_hist_%s",  datasetNames_.at(a).c_str());

    TString path_ORIG_title = "Total Times Path Fires ORIG (" + datasetNames_.at(a) + " dataset)";
    TString path_ORIGnotNEW_title = "Path fires in ORIG but not in NEW (" + datasetNames_.at(a) + " dataset)";
    TString path_NEWnotORIG_title = "Path fires in NEW but not in ORIG (" + datasetNames_.at(a) + " dataset)";
    TString pathmodule_ORIGnotNEW_title = "Last run module index vs Path for NEW, when ORIG fired but NEW didn't (" + datasetNames_.at(a) + " dataset)";
    TString pathmodule_NEWnotORIG_title = "Last run module index vs Path for ORIG, when NEW fired but ORIG didn't (" + datasetNames_.at(a) + " dataset)";

    path_ORIG_hist_PD.push_back(dqms_->book1D(path_ORIG_name,path_ORIG_title,nPaths_PD_.at(a),0,nPaths_PD_.at(a)));
    path_ORIGnotNEW_hist_PD.push_back(dqms_->book1D(path_ORIGnotNEW_name,path_ORIGnotNEW_title,nPaths_PD_.at(a),0,nPaths_PD_.at(a)));
    path_NEWnotORIG_hist_PD.push_back(dqms_->book1D(path_NEWnotORIG_name,path_NEWnotORIG_title,nPaths_PD_.at(a),0,nPaths_PD_.at(a)));
    pathmodule_ORIGnotNEW_hist_PD.push_back(dqms_->book2D(pathmodule_ORIGnotNEW_name,pathmodule_ORIGnotNEW_title,nPaths_PD_.at(a),0,nPaths_PD_.at(a),max_nModules_PD_.at(a),0,max_nModules_PD_.at(a)));
    pathmodule_NEWnotORIG_hist_PD.push_back(dqms_->book2D(pathmodule_NEWnotORIG_name,pathmodule_NEWnotORIG_title,nPaths_PD_.at(a),0,nPaths_PD_.at(a),max_nModules_PD_.at(a),0,max_nModules_PD_.at(a)));
  }
  
}

// ------------ method called when ending the processing of a run  ------------
void 
HLTOfflineReproducibilityDQM::endRun(edm::Run const&, edm::EventSetup const&)
{
  //all paths
  cout<<endl;
  for (unsigned int i=0; i<nPaths_; i++) {
    cout<<triggerNames_.at(i)<<" ORIG accepts: "<<trigger_ORIG_.at(i)<<", NEW accepts: "<<trigger_NEW_.at(i)<<endl;
    path_ORIG_hist->getTH1F()->GetXaxis()->SetBinLabel(i+1,triggerNames_.at(i).c_str());
    path_ORIGnotNEW_hist->getTH1F()->GetXaxis()->SetBinLabel(i+1,triggerNames_.at(i).c_str());
    path_NEWnotORIG_hist->getTH1F()->GetXaxis()->SetBinLabel(i+1,triggerNames_.at(i).c_str());
    pathmodule_ORIGnotNEW_hist->getTH2F()->GetXaxis()->SetBinLabel(i+1,triggerNames_.at(i).c_str());
    pathmodule_NEWnotORIG_hist->getTH2F()->GetXaxis()->SetBinLabel(i+1,triggerNames_.at(i).c_str());
  }

  //paths per dataset
  for (unsigned int a=0; a<nDatasets_; a++) {
    for (unsigned int b=0; b<nPaths_PD_.at(a); b++) {
      path_ORIG_hist_PD.at(a)->getTH1F()->GetXaxis()->SetBinLabel(b+1,datasetContent_.at(a).at(b).c_str());
      path_ORIGnotNEW_hist_PD.at(a)->getTH1F()->GetXaxis()->SetBinLabel(b+1,datasetContent_.at(a).at(b).c_str());
      path_NEWnotORIG_hist_PD.at(a)->getTH1F()->GetXaxis()->SetBinLabel(b+1,datasetContent_.at(a).at(b).c_str());
      pathmodule_ORIGnotNEW_hist_PD.at(a)->getTH2F()->GetXaxis()->SetBinLabel(b+1,datasetContent_.at(a).at(b).c_str());
      pathmodule_NEWnotORIG_hist_PD.at(a)->getTH2F()->GetXaxis()->SetBinLabel(b+1,datasetContent_.at(a).at(b).c_str());
    }
  }

}

// ------------ method called when starting to processes a luminosity block  ------------
void 
HLTOfflineReproducibilityDQM::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
HLTOfflineReproducibilityDQM::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HLTOfflineReproducibilityDQM::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTOfflineReproducibilityDQM);
