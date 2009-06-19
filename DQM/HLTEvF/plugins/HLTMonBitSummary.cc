//HLTMonBitSummary.cc
//combination of HLTMon.cc and UserCode/AEverett/BitPlotting

// user include files
#include "DQM/HLTEvF/interface/HLTMonBitSummary.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Run.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h" 
#include "TH1F.h"
#include "TProfile.h"
#include "TH2F.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;
using namespace std;

HLTMonBitSummary::HLTMonBitSummary(const edm::ParameterSet& iConfig) :
  inputTag_ (iConfig.getParameter<edm::InputTag> ("TriggerResultsTag")),
  HLTPathsByName_(iConfig.getParameter<std::vector<std::string > >("HLTPaths")),
  filterTypes_(iConfig.getParameter<std::vector<std::string > >("filterTypes")),
  total_(0),
  nValidTriggers_(0)
{
  denominator_ = iConfig.getUntrackedParameter<std::string>("denominator");
  directory_ = iConfig.getUntrackedParameter<std::string>("directory","HLT/HLTMonMuon");
  //label_ = iConfig.getParameter<std::string>("label");
  //  out_ = iConfig.getUntrackedParameter<std::string>("out","");

  dbe_ = NULL;
  dbe_ = Service < DQMStore > ().operator->();
  dbe_->setVerbose(0);
  
  //initialize the hlt configuration from the process name if not blank
  std::string processName = inputTag_.process();
  if (processName != ""){
    std::stringstream buffer;
    //get the configuration
    HLTConfigProvider hltConfig;
    hltConfig.init(processName);
    std::vector<std::string> validTriggerNames = hltConfig.triggerNames();
  
    if (validTriggerNames.size() < 1) {
      LogDebug ("HLTMonBitSummary") << endl << endl << endl
				    << "---> WARNING: HLTConfigProvider has empty list of valid trigger names" << endl
				    << "Check the HLT Process Name (you provided  " << processName <<")" << endl
				    << "W/o valid triggers we can't produce plots, exiting..."
				    << endl << endl << endl;
      return;
    }
  
          
    if (validTriggerNames.size()!=0){
      bool goodToGo = false;
      //remove all path names that are not valid
      while(!goodToGo && HLTPathsByName_.size()!=0){
	goodToGo=true;
	for (std::vector<std::string>::iterator j=HLTPathsByName_.begin();j!=HLTPathsByName_.end();++j){
	  bool goodOne = false;
	  //check if trigger name is valid
	  for (unsigned int i = 0; i != validTriggerNames.size(); ++i){
	    if (validTriggerNames[i]==(*j)){ goodOne = true; break; }
	  }
	  if (!goodOne){
	    goodToGo = false;
	    buffer << (*j) << " is not a valid trigger in process: " << processName << std::endl;
	    HLTPathsByName_.erase(j);
	    break;
	  }
	}
      }
      LogDebug("HLTMonBitSummary|BitStatus")<<buffer.str();
    }
    
    
    count_.resize(HLTPathsByName_.size());
    HLTPathsByIndex_.resize(HLTPathsByName_.size());
    
    vector<string>::const_iterator iDumpName;
    unsigned int numTriggers = 0;
    for (iDumpName = validTriggerNames.begin(); iDumpName != validTriggerNames.end(); iDumpName++) {
      LogDebug ("HLTMonBitSummary") << "Trigger " << numTriggers   
				    << " is called " << (*iDumpName)
				    << endl;
      numTriggers++;
    }
    
        
    nValidTriggers_ = HLTPathsByName_.size();
    
    for( size_t i = 0; i < nValidTriggers_; i++) {
      // create a row [triggername,filter1name, filter2name, etc.] 
      triggerFilters_.push_back(vector <string>());  
      // create a row [0, filter1index, filter2index, etc.]
      triggerFilterIndices_.push_back(vector <uint>()); 
      
      vector<string> moduleNames = hltConfig.moduleLabels( HLTPathsByName_[i] ); 
            
      //triggerFilters_[nValidTriggers_-1].push_back(HLTPathsByName_[i]);//first entry is trigger name 
      triggerFilters_[i].push_back(HLTPathsByName_[i]);//first entry is trigger name
	      
      //triggerFilterIndices_[nValidTriggers_-1].push_back(0);
      triggerFilterIndices_[i].push_back(0);
      
      int numModule = 0, numFilters = 0;
      string moduleName, moduleType;
      unsigned int moduleIndex;
      
      //print module name
      
      vector<string>::const_iterator iDumpModName;
      for (iDumpModName = moduleNames.begin();iDumpModName != moduleNames.end();iDumpModName++) {
	moduleName = *iDumpModName;
	moduleType = hltConfig.moduleType(moduleName);
	moduleIndex = hltConfig.moduleIndex(HLTPathsByName_[i], moduleName);
	LogDebug ("HLTMonBitSummary") << "Module"      << numModule
				      << " is called " << moduleName
				      << " , type = "  << moduleType
				      << " , index = " << moduleIndex
				      << endl;
	numModule++;
	for(size_t k = 0; k < filterTypes_.size(); k++) {
	  if(moduleType == filterTypes_[k]) {
	    numFilters++;
	    //triggerFilters_[nValidTriggers_-1].push_back(moduleName);
	    triggerFilters_[i].push_back(moduleName);
	    //triggerFilterIndices_[nValidTriggers_-1].push_back(moduleIndex);
	    triggerFilterIndices_[i].push_back(moduleIndex);
	  }
	}
      }//end for modulesName
      
    }//end for nValidTriggers_
    
  }//end if processName
    
}


HLTMonBitSummary::~HLTMonBitSummary(){}

//
// member functions
//

void HLTMonBitSummary::beginRun(const edm::Run  & r, const edm::EventSetup  &){
    
  if(dbe_){
    if (directory_ != "" ) directory_ = directory_+"/" ;

    int nbin = nValidTriggers_;

    dbe_->setCurrentFolder(directory_ + "Summary/");

    //int nbin_sub = 5;
    int nbin_sub = 8;

    // Count histos for efficiency plots
    dbe_->setCurrentFolder(directory_ + "Summary/CountHistos/");
    //hCountSummary = dbe_->book1D("hCountSummary", "Count Summary", nbin+1, -0.5, 0.5+(double)nbin);
    
    for( int trig = 0; trig < nbin; trig++ ) {
      // count plots for subfilter
      hSubFilterCount[trig] = dbe_->book1D("Filters_" + triggerFilters_[trig][0], 
    					   "Filters_" + triggerFilters_[trig][0],
					   nbin_sub+1, -0.5, 0.5+(double)nbin_sub);
      
      for(int filt = 0; filt < (int)triggerFilters_[trig].size()-1; filt++){
      	hSubFilterCount[trig]->setBinLabel(filt+1, triggerFilters_[trig][filt+1]);
      }
    }


    //--------------B i t   P l o t t i n g   s e c t i o n ---------------//
    //---------------------------------------------------------------------//

    std::stringstream rNs;
    rNs<<r.run();
    std::string rN = rNs.str();
  
    float min = -0.5;
    float max = HLTPathsByName_.size()-0.5;
    uint nBin = HLTPathsByName_.size();
  
    LogDebug("HLTMonBitSummary")<<"this is the beginning of a NEW run: "<< r.run();
 
    dbe_->setCurrentFolder(directory_+"Summary");

    //h1_ = edm::Service<DQMStore>()->book1D(std::string("passingBits_"+label_+"_"+rN),std::string("passing bits in run "+rN),nBin,min,max);  
    h1_ = dbe_->book1D("PassingBits_Summary","PassingBits_Summary", nBin, min, max);
    //h2_ = edm::Service<DQMStore>()->book2D(std::string("passingBits_2_"+label_+"_"+rN),std::string("correlation between bits in run "+rN),nBin,min,max, nBin,min,max);  
    h2_ = dbe_->book2D("PassingBits_Correlation","PassingBits_Correlation",nBin,min,max, nBin,min,max);
    //pf_ = edm::Service<DQMStore>()->bookProfile(std::string("fraction_"+label_+"_"+rN),std::string("fraction of passing bits in run "+rN),nBin,min,max, 1000, 0.0, 1.0);  
    pf_ = dbe_->bookProfile("Efficiency_Summary","Efficiency_Summary", nBin, min, max, 1000, 0.0, 1.0);
    if (denominator_!="")
      //ratio_ = edm::Service<DQMStore>()->bookProfile(std::string("ratio_"+label_+"_"+rN),std::string("fraction of passing bits in run "+rN+" with respect to: "+denominator_),nBin,min,max, 1000, 0.0, 1.0);
      ratio_ = dbe_->bookProfile(std::string("Ratio_"+denominator_),std::string("Ratio_"+denominator_),nBin,min,max, 1000, 0.0, 1.0);
    else 
      ratio_=0;

    for (uint i=0; i!=nBin; ++i){
      h1_->getTH1F()->GetXaxis()->SetBinLabel(i+1,HLTPathsByName_[i].c_str());
      h2_->getTH2F()->GetXaxis()->SetBinLabel(i+1,HLTPathsByName_[i].c_str());
      h2_->getTH2F()->GetYaxis()->SetBinLabel(i+1,HLTPathsByName_[i].c_str());
      pf_->getTProfile()->GetXaxis()->SetBinLabel(i+1,HLTPathsByName_[i].c_str());
      if (ratio_)
	ratio_->getTProfile()->GetXaxis()->SetBinLabel(i+1,(HLTPathsByName_[i]+" & "+denominator_).c_str());
    }

    //------------------------End Of BitPlotting section -------------------------//
  }
  
}


// ------------ method called to for each event  ------------
void
HLTMonBitSummary::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    
  const string invalid("@@invalid@@");

  // get hold of TriggerResults Object
  Handle<TriggerResults> trh;
  iEvent.getByLabel(inputTag_,trh);
  
  if (trh.failedToGet()) {
     edm::LogError("HLTMonBitSummary")<<" could not get: "<<inputTag_;
     return;
  }

  // get hold of trigger names - based on TriggerResults object!
  triggerNames_.init(*trh);
  
  unsigned int lastModule = 0;
  //unsigned int n = HLTPathsByName_.size();
  //convert trigger names to trigger index properly
  for (unsigned int trig=0; trig < nValidTriggers_; trig++) {
    HLTPathsByIndex_[trig]=triggerNames_.triggerIndex(HLTPathsByName_[trig]);
    lastModule = trh->index(HLTPathsByIndex_[trig]);
    //cout << "Trigger Name = " << HLTPathsByName_[trig] << ", HLTPathsByIndex_ = " << HLTPathsByIndex_[trig] << endl; 
    //cout << "Trigger Name = " << HLTPathsByName_[trig] << ", trh->index = " << lastModule << endl; 
 
    //go through the list of filters
    for(unsigned int filt = 0; filt < triggerFilters_[trig].size()-1; filt++){
      if(triggerFilterIndices_[trig][filt+1] <= lastModule){//check if filter passed
	//cout << "triggerFilters_["<<trig<<"]["<<filt+1<<"] = " << triggerFilters_[trig][filt+1] 
	//     << " , triggerFilterIndices = " << triggerFilterIndices_[trig][filt+1]
	//     << " , lastModule = " << lastModule << endl;
	if(hSubFilterCount[trig]){
	  int binNumber = hSubFilterCount[trig]->getTH1F()->GetXaxis()->FindBin(triggerFilters_[trig][filt+1].c_str());
	  hSubFilterCount[trig]->Fill(binNumber-1);
	}
      }
    }

  }

  //and check validity name (should not be necessary)
  std::vector<bool> validity(nValidTriggers_);
  for (unsigned int i=0; i!=nValidTriggers_; i++) {
    validity[i]=( (HLTPathsByIndex_[i]<trh->size()) && (HLTPathsByName_[i]!=invalid) );
  }
   
  
  //---------------------B i t   P l o t t i n g   S e c t i o n --------------------//
  //---------------------------------------------------------------------------------//
  
  //convert also for the denominator and check validity
  uint denominatorIndex = 0;
  bool denominatorValidity= false;
  if (denominator_!="") {
    denominatorIndex=triggerNames_.triggerIndex(denominator_);
    denominatorValidity= (denominatorIndex <trh->size());
  }
  
  std::stringstream report;
  std::string sep=" ";
  bool atLeasOne=false;
  
  //check whether the denominator fired
  bool denomAccept=false;
  if (ratio_ && denominatorValidity) denomAccept=trh->accept(denominatorIndex);
  
  for (unsigned int i=0; i!=nValidTriggers_; i++) {
    if (!validity[i]) continue;
    bool iAccept=trh->accept(HLTPathsByIndex_[i]);
    if (iAccept) {
      report<<sep<<HLTPathsByName_[i];
      count_[i]++;
      sep=", ";
      atLeasOne=true;
      //trigger has fired. make an entry in both 1D and profile plots
      h1_->Fill(i);
      pf_->Fill(i,1);
      //make the entry in the 2D plot : UPPER diagonal terms = AND of the two triggers
      for (unsigned int j=i; j!=nValidTriggers_; j++) {
	if (!validity[j]) continue;
	if (trh->accept(HLTPathsByIndex_[j]))
	  h2_->Fill(i,j);    
      }//loop on second trigger for AND terms
    }//trigger[i]=true
       
    else{
       //make an entry at zero to the profile
      pf_->Fill(i,0);
    }//trigger[i]=false
    
     //make proper entries in the ratio plot
    if (ratio_ && denomAccept){
      if (iAccept) ratio_->Fill(i,1);
      else ratio_->Fill(i,0);
    }
    
    //make proper entry inthe 2D plot: LOWER diagonal terms = OR of the two triggers
    for (unsigned int j=0; j!=i; j++) {
      if (!validity[j]) continue;
      bool jAccept=trh->accept(HLTPathsByIndex_[j]);
      if (iAccept || jAccept)
	h2_->Fill(i,j);
     }//loop on second trigger for OR terms
    
  }//loop on first trigger

  if (atLeasOne){
    LogDebug("BitPlotting|BitReport")<<report.str();
  }
  
  total_++;
  
  //   edm::LogError("BitPlotting")<<"# entries:"<<h1_->getTH1F()->GetEntries();
  
  //-----------------E n d  o f  B i t P l o t t i n g   S e c t i o n-----------------//
}


// ------------ method called once each job just after ending the event loop  ------------
void HLTMonBitSummary::endJob() {  
 
  std::stringstream report;
  report <<" out of: "<<total_<<" events.\n";
  for (uint i=0; i!=HLTPathsByName_.size();i++){
    report<<HLTPathsByName_[i]<<" passed: "<<count_[i]<<" times.\n";
    count_[i]=0;
  }
  
  edm::LogInfo("HLTMonBitSummary|BitSummary")<<report.str();
  LogDebug("HLTMonBitSummary|BitSummary")<<report.str();
  total_=0;
  //  if( out_.size() != 0 ) edm::Service<DQMStore>()->save(out_);
  
}

