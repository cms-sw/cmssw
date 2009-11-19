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
#include "DQM/HLTEvF/interface/HLTriggerSelector.h"

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
  nValidTriggers_(0),
  ndenomAccept_(0)
{
  denominatorWild_ = iConfig.getUntrackedParameter<std::string>("denominatorWild","");
  denominator_ = iConfig.getUntrackedParameter<std::string>("denominator");
  directory_ = iConfig.getUntrackedParameter<std::string>("directory","HLT/HLTMonMuon");
  histLabel_ = iConfig.getUntrackedParameter<std::string>("histLabel","Muon");
  //label_ = iConfig.getParameter<std::string>("label");
  //  out_ = iConfig.getUntrackedParameter<std::string>("out","");

  dbe_ = NULL;
  dbe_ = Service < DQMStore > ().operator->();
  dbe_->setVerbose(0);

}


HLTMonBitSummary::~HLTMonBitSummary(){}

//
// member functions
//

void HLTMonBitSummary::beginRun(const edm::Run  & r, const edm::EventSetup  &){
    
  //initialize the hlt configuration from the process name if not blank
  std::string processName = inputTag_.process();
  if (processName != ""){
    //get the configuration
    HLTConfigProvider hltConfig;
    hltConfig.init(processName);
  
    //run trigger selection
    HLTriggerSelector trigSelect(inputTag_,HLTPathsByName_);
    HLTPathsByName_.swap(trigSelect.theSelectTriggers);
    count_.resize(HLTPathsByName_.size());
    HLTPathsByIndex_.resize(HLTPathsByName_.size());
    
    nValidTriggers_ = HLTPathsByName_.size();
    
    //get all the filters
    for( size_t i = 0; i < nValidTriggers_; i++) {
      // create a row [triggername,filter1name, filter2name, etc.] 
      triggerFilters_.push_back(vector <string>());  
      // create a row [0, filter1index, filter2index, etc.]
      triggerFilterIndices_.push_back(vector <uint>()); 
      
      vector<string> moduleNames = hltConfig.moduleLabels( HLTPathsByName_[i] ); 
      
      triggerFilters_[i].push_back(HLTPathsByName_[i]);//first entry is trigger name      
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
	    triggerFilters_[i].push_back(moduleName);
	    triggerFilterIndices_[i].push_back(moduleIndex);
	  }
	}
      }//end for modulesName
    }//end for nValidTriggers_


    //check denominator
    if( denominatorWild_.size() != 0 ) HLTPathDenomName_.push_back(denominatorWild_);
    HLTriggerSelector denomSelect(inputTag_,HLTPathDenomName_);
    HLTPathDenomName_.swap(denomSelect.theSelectTriggers);
    //for (unsigned int i = 0; i < HLTPathDenomName_.size(); i++)
    //  std::cout << "testing denom: " << HLTPathDenomName_[i] << std::endl;
    if(HLTPathDenomName_.size()==1) denominator_ = HLTPathDenomName_[0];

  }//end if process


  if(dbe_){

    if (directory_ != "" && directory_.substr(directory_.length()-1,1) != "/" ) directory_ = directory_+"/" ;

    int nbin = nValidTriggers_;

    dbe_->setCurrentFolder(directory_ + "Summary/");

    //int nbin_sub = 5;
    int nbin_sub = 8;

    // Count histos for efficiency plots
    dbe_->setCurrentFolder(directory_ + "Summary/Trigger_Filters/");
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

    dbe_->setCurrentFolder(directory_ + "Summary/Trigger_Efficiencies/");
    for( int trig = 0; trig < nbin; trig++ ) {
      hSubFilterEff[trig] = dbe_->book1D("Efficiency_" + triggerFilters_[trig][0], 
					 "Efficiency_" + triggerFilters_[trig][0],
					 nbin_sub+1, -0.5, 0.5+(double)nbin_sub);
      
      for(int filt = 0; filt < (int)triggerFilters_[trig].size()-1; filt++){
	hSubFilterEff[trig]->setBinLabel(filt+1,triggerFilters_[trig][filt+1]);
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

    h1_ = dbe_->book1D("PassingBits_Summary_"+histLabel_,"PassingBits_Summary_"+histLabel_, nBin, min, max);
    h2_ = dbe_->book2D("PassingBits_Correlation_"+histLabel_,"PassingBits_Correlation_"+histLabel_,nBin,min,max, nBin,min,max);
    pf_ = dbe_->book1D("Efficiency_Summary_"+histLabel_,"Efficiency_Summary_"+histLabel_, nBin, min, max);
    if (denominator_!="")
      //ratio_ = dbe_->book1D(std::string("Ratio_"+denominator_),std::string("Ratio_"+denominator_),nBin,min,max);
      ratio_ = dbe_->book1D("HLTRate_"+histLabel_,"HLTRate_"+histLabel_,nBin,min,max);
    else 
      ratio_=0;

    for (uint i=0; i!=nBin; ++i){
      h1_->getTH1F()->GetXaxis()->SetBinLabel(i+1,HLTPathsByName_[i].c_str());
      h2_->getTH2F()->GetXaxis()->SetBinLabel(i+1,HLTPathsByName_[i].c_str());
      h2_->getTH2F()->GetYaxis()->SetBinLabel(i+1,HLTPathsByName_[i].c_str());
      pf_->getTH1F()->GetXaxis()->SetBinLabel(i+1,HLTPathsByName_[i].c_str());
      if (ratio_)
	ratio_->getTH1F()->GetXaxis()->SetBinLabel(i+1,(HLTPathsByName_[i]+" & "+denominator_).c_str());
    }

    //------------------------End Of BitPlotting section -------------------------//
  }
  
}


// ------------ method called to for each event  ------------
void
HLTMonBitSummary::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  total_++;    
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

  //convert trigger names to trigger index properly
  for (unsigned int trig=0; trig < nValidTriggers_; trig++) {
    HLTPathsByIndex_[trig]=triggerNames_.triggerIndex(HLTPathsByName_[trig]);
    lastModule = trh->index(HLTPathsByIndex_[trig]);
    //cout << "Trigger Name = " << HLTPathsByName_[trig] << ", HLTPathsByIndex_ = " << HLTPathsByIndex_[trig] << endl; 
    //cout << "Trigger Name = " << HLTPathsByName_[trig] << ", trh->index = " << lastModule << " " << trh->accept(HLTPathsByIndex_[trig]) << endl; 
  
    //go through the list of filters
    for(unsigned int filt = 0; filt < triggerFilters_[trig].size()-1; filt++){
      // 	cout << "triggerFilters_["<<trig<<"]["<<filt+1<<"] = " << triggerFilters_[trig][filt+1] 
      // 	     << " , triggerFilterIndices = " << triggerFilterIndices_[trig][filt+1]
      // 	     << " , lastModule = " << lastModule << endl;
      
      int binNumber = hSubFilterCount[trig]->getTH1F()->GetXaxis()->FindBin(triggerFilters_[trig][filt+1].c_str());      
      
      //check if filter passed
      if(trh->accept(HLTPathsByIndex_[trig])){
	hSubFilterCount[trig]->Fill(binNumber-1);//binNumber1 = 0 = first filter
      }
      //otherwise the module that issued the decision is the first fail
      //so that all the ones before it passed
      else if(triggerFilterIndices_[trig][filt+1] < lastModule){
	hSubFilterCount[trig]->Fill(binNumber-1);
      }
      
      //hSubFilterCount[trig]->Fill(-1);
      
      float eff = (float)hSubFilterCount[trig]->getBinContent(binNumber) / (float)total_ ;
      float efferr = sqrt(eff*(1-eff)/ (float)total_);
      hSubFilterEff[trig]->setBinContent(binNumber,eff);
      hSubFilterEff[trig]->setBinError(binNumber,efferr);

    }//filt
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
  bool atLeastOne=false;
  
  //check whether the denominator fired
  bool denomAccept=false;
  if (ratio_ && denominatorValidity) {
    denomAccept=trh->accept(denominatorIndex);
    if(denomAccept) ndenomAccept_++;
  }

  for (unsigned int i=0; i!=nValidTriggers_; i++) {
    if (!validity[i]) continue;
    bool iAccept=trh->accept(HLTPathsByIndex_[i]);
    if (iAccept) {
      report<<sep<<HLTPathsByName_[i];
      count_[i]++;
      sep=", ";
      atLeastOne=true;
      //trigger has fired. make an entry in both 1D and profile plots
      h1_->Fill(i);
      //make the entry in the 2D plot : UPPER diagonal terms = AND of the two triggers
      for (unsigned int j=i; j!=nValidTriggers_; j++) {
	if (!validity[j]) continue;
	if (trh->accept(HLTPathsByIndex_[j]))
	  h2_->Fill(i,j);    
      }//loop on second trigger for AND terms
    }//trigger[i]=true
       
    float pf_eff = (float)h1_->getBinContent(i+1) / (float)total_ ;
    float pf_efferr = sqrt(pf_eff*(1-pf_eff) / (float)total_);
    pf_->setBinContent(i+1,pf_eff);
    pf_->setBinError(i+1,pf_efferr);

    //make proper entries in the ratio plot
    if (ratio_ && denomAccept){
      float ratio_eff = (float)h1_->getBinContent(i+1) / (float)ndenomAccept_ ;
      float ratio_efferr = sqrt(ratio_eff*(1-ratio_eff) / (float)ndenomAccept_);
      ratio_->setBinContent(i+1,ratio_eff);
      ratio_->setBinError(i+1,ratio_efferr);
    }

    
    //make proper entry inthe 2D plot: LOWER diagonal terms = OR of the two triggers
    for (unsigned int j=0; j!=i; j++) {
      if (!validity[j]) continue;
      bool jAccept=trh->accept(HLTPathsByIndex_[j]);
      if (iAccept || jAccept)
	h2_->Fill(i,j);
     }//loop on second trigger for OR terms
    
  }//loop on first trigger

  if (atLeastOne){
    LogDebug("BitPlotting|BitReport")<<report.str();
  }
  
  //   edm::LogError("BitPlotting")<<"# entries:"<<h1_->getTH1F()->GetEntries();
  
  //-----------------E n d  o f  B i t P l o t t i n g   S e c t i o n-----------------//
  
}


// ------------ method called once each job just after ending the event loop  ------------
void HLTMonBitSummary::endJob() {  
  
  std::stringstream report;
  report <<" out of: "<<total_<<" events.\n";
  if(!count_.empty()){
    for (uint i=0; i!=HLTPathsByName_.size();i++){
      report<<HLTPathsByName_[i]<<" passed: "<<count_[i]<<" times.\n";
      count_[i]=0;
    }
  }
  
  edm::LogInfo("HLTMonBitSummary|BitSummary")<<report.str();
  LogDebug("HLTMonBitSummary|BitSummary")<<report.str();
  total_=0;
  //  if( out_.size() != 0 ) edm::Service<DQMStore>()->save(out_);
  
}

