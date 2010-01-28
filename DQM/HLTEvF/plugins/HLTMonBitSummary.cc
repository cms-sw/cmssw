//HLTMonBitSummary.cc
//combination of HLTMon.cc and UserCode/AEverett/BitPlotting

// user include files
#include "DQM/HLTEvF/interface/HLTMonBitSummary.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Run.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DQM/HLTEvF/interface/HLTriggerSelector.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h" 
#include "TH1F.h"
#include "TProfile.h"
#include "TH2F.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// needed for trigger bits from EventSetup as in ALCARECO paths
#include "CondFormats/HLTObjects/interface/AlCaRecoTriggerBits.h"
#include "CondFormats/DataRecord/interface/AlCaRecoTriggerBitsRcd.h"

using namespace edm;
using namespace std;

HLTMonBitSummary::HLTMonBitSummary(const edm::ParameterSet& iConfig) :
  inputTag_ (iConfig.getParameter<edm::InputTag> ("TriggerResultsTag")),
  HLTPathNamesConfigPreVal_(iConfig.getParameter<std::vector<std::string > >("HLTPaths")),
  total_(0),
  nValidTriggers_(0),
  nValidConfigTriggers_(0),
  ndenomAccept_(0)
{
  denominatorWild_ = iConfig.getUntrackedParameter<std::string>("denominatorWild","");
  denominator_ = iConfig.getUntrackedParameter<std::string>("denominator");
  directory_ = iConfig.getUntrackedParameter<std::string>("directory","HLT/Generic/Summary");
  histLabel_ = iConfig.getUntrackedParameter<std::string>("histLabel","Generic");
  //  out_ = iConfig.getUntrackedParameter<std::string>("out","");
  dummyFilters_.clear();
  filterTypes_ = iConfig.getUntrackedParameter<std::vector<std::string > >("filterTypes",dummyFilters_);
  esPathsKey_ = iConfig.getUntrackedParameter<std::string>("eventSetupPathsKey","");

  configFlag_ = false;
  filterFlag_ = false;

  dbe_ = NULL;
  dbe_ = Service < DQMStore > ().operator->();
  dbe_->setVerbose(0);

}


HLTMonBitSummary::~HLTMonBitSummary(){}

//
// member functions
//

void HLTMonBitSummary::beginRun(const edm::Run  & r, const edm::EventSetup  &iSetup){

  //initialize the hlt configuration from the process name if not blank
  std::string processName = inputTag_.process();
  if (processName != ""){

    //Grab paths from EventSetup via AlCaRecoTriggerBitsRcd if configured - copied from HLTHighLevel
    if (esPathsKey_.size()) {
      // Get map of strings to concatenated list of names of HLT paths from EventSetup:
      edm::ESHandle<AlCaRecoTriggerBits> triggerBits;
      iSetup.get<AlCaRecoTriggerBitsRcd>().get(triggerBits);
      typedef std::map<std::string, std::string> TriggerMap;
      const TriggerMap &triggerMap = triggerBits->m_alcarecoToTrig;

      TriggerMap::const_iterator listIter = triggerMap.find(esPathsKey_);
    
      if (listIter == triggerMap.end()) {
	throw cms::Exception("Configuration")
	  //<< " HLTHighLevel [instance: " << *moduleLabel() << " - path: " << *pathName()
	  //<< "]: "
	  <<" No triggerList with key " << esPathsKey_ << " in AlCaRecoTriggerBitsRcd";
      }
      
      // We must avoid a map<string,vector<string> > in DB for performance reason,
      // so the paths are mapped into one string that we have to decompose:
      HLTPathNamesKey_ = triggerBits->decompose(listIter->second);
    }
    //otherwise read HLTPaths from configuration
    HLTPathNamesConfig_.clear();
    if(HLTPathNamesConfigPreVal_.size()){
      //run trigger selection
      HLTriggerSelector trigSelect(inputTag_,HLTPathNamesConfigPreVal_);
      HLTPathNamesConfig_.swap(trigSelect.theSelectTriggers);
    }

    //check if the two vectors have any common elements    
    vector<int> removePaths;
    for(size_t i=0; i<HLTPathNamesKey_.size(); ++i){
      for(size_t j=0; j<HLTPathNamesConfig_.size(); ++j){
	if(HLTPathNamesConfig_[j] == HLTPathNamesKey_[i]) removePaths.push_back(i); 
      }
    }
    reverse(removePaths.begin(),removePaths.end());    
    if(removePaths.size()){
      for(unsigned int k=0; k<removePaths.size(); ++k)
	HLTPathNamesKey_.erase(HLTPathNamesKey_.begin()+removePaths[k]);
    }


    //combine two vectors
    HLTPathsByName_.clear();
    HLTPathsByName_.reserve(HLTPathNamesConfig_.size() + HLTPathNamesKey_.size());
    HLTPathsByName_.insert(HLTPathsByName_.end(),HLTPathNamesConfig_.begin(),HLTPathNamesConfig_.end());
    HLTPathsByName_.insert(HLTPathsByName_.end(),HLTPathNamesKey_.begin(),HLTPathNamesKey_.end());

    count_.resize(HLTPathsByName_.size());
    HLTPathsByIndex_.resize(HLTPathsByName_.size());

    
    if(nValidTriggers_ != HLTPathsByName_.size() && total_!=0 ){
      LogWarning("HLTMonBitSummary") << "The number of valid triggers has changed since beginning of job." 
				     << std::endl
				     << "BitSummary histograms do not support changing configurations."
				     << std::endl
				     << "Processing of events halted.";
      configFlag_ = true;
    }

    if(!configFlag_){

      nValidTriggers_ = HLTPathsByName_.size();
      nValidConfigTriggers_ = HLTPathNamesConfig_.size();

      //get the configuration
      HLTConfigProvider hltConfig;
  
      //get all the filters -
      //only if filterTypes_ is nonempty and only on HLTPathNamesConfig_ paths
      if( hltConfig.init(processName)){ 
	if(!filterTypes_.empty()){
	  triggerFilters_.clear();
	  triggerFilterIndices_.clear();
	  for( size_t i = 0; i < nValidConfigTriggers_; i++) {
	    // create a row [triggername,filter1name, filter2name, etc.] 
	    triggerFilters_.push_back(vector <string>());  
	    // create a row [0, filter1index, filter2index, etc.]
	    triggerFilterIndices_.push_back(vector <uint>()); 
      
	    vector<string> moduleNames = hltConfig.moduleLabels( HLTPathNamesConfig_[i] ); 
      
	    triggerFilters_[i].push_back(HLTPathNamesConfig_[i]);//first entry is trigger name      
	    triggerFilterIndices_[i].push_back(0);
      
	    int numModule = 0, numFilters = 0;
	    string moduleName, moduleType;
	    unsigned int moduleIndex;
      
	    //print module name
	    vector<string>::const_iterator iDumpModName;
	    for (iDumpModName = moduleNames.begin();iDumpModName != moduleNames.end();iDumpModName++) {
	      moduleName = *iDumpModName;
	      moduleType = hltConfig.moduleType(moduleName);
	      moduleIndex = hltConfig.moduleIndex(HLTPathNamesConfig_[i], moduleName);
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
	  }//end for nValidConfigTriggers_
	}
      }
      else{
	LogError("HLTMonBitSummary") << "HLTConfigProvider initialization with process name " 
				       << processName << " failed." << endl
				       << "Could not get filter names." << endl;
	filterFlag_ = true;
      }

      //check denominator
      HLTPathDenomName_.clear();
      if( denominatorWild_.size() != 0 ) HLTPathDenomName_.push_back(denominatorWild_);
      HLTriggerSelector denomSelect(inputTag_,HLTPathDenomName_);
      HLTPathDenomName_.swap(denomSelect.theSelectTriggers);
      //for (unsigned int i = 0; i < HLTPathDenomName_.size(); i++)
      //  std::cout << "testing denom: " << HLTPathDenomName_[i] << std::endl;
      if(HLTPathDenomName_.size()==1) denominator_ = HLTPathDenomName_[0];
    }
  }//end if process

       
  if(dbe_ && !configFlag_){

    int nbin = nValidConfigTriggers_;
    
    dbe_->setCurrentFolder(directory_);

    //int nbin_sub = 5;
    int nbin_sub = 8;
    
    // Count histos for efficiency plots
    if(!filterTypes_.empty() && !filterFlag_){
      dbe_->setCurrentFolder(directory_ + "Trigger_Filters/");
      //hCountSummary = dbe_->book1D("hCountSummary", "Count Summary", nbin+1, -0.5, 0.5+(double)nbin);
    
      hSubFilterCount.clear();
      hSubFilterEff.clear();

      for( int trig = 0; trig < nbin; trig++ ) {
	// count plots for subfilter
	//hSubFilterCount[trig] = dbe_->book1D("Filters_" + triggerFilters_[trig][0], 
	hSubFilterCount.push_back(dbe_->book1D("Filters_" + triggerFilters_[trig][0], 
					     "Filters_" + triggerFilters_[trig][0],
					     nbin_sub+1, -0.5, 0.5+(double)nbin_sub));
      
	for(int filt = 0; filt < (int)triggerFilters_[trig].size()-1; filt++){
	  hSubFilterCount[trig]->setBinLabel(filt+1, triggerFilters_[trig][filt+1]);
	}
      }

      dbe_->setCurrentFolder(directory_ + "Trigger_Efficiencies/");
      for( int trig = 0; trig < nbin; trig++ ) {
	//hSubFilterEff[trig] = dbe_->book1D("Efficiency_" + triggerFilters_[trig][0], 
	hSubFilterEff.push_back(dbe_->book1D("Efficiency_" + triggerFilters_[trig][0], 
					   "Efficiency_" + triggerFilters_[trig][0],
					   nbin_sub+1, -0.5, 0.5+(double)nbin_sub));
      
	for(int filt = 0; filt < (int)triggerFilters_[trig].size()-1; filt++){
	  hSubFilterEff[trig]->setBinLabel(filt+1,triggerFilters_[trig][filt+1]);
	}
      }
    }
      
    //--------------B i t   P l o t t i n g   s e c t i o n ---------------//
    //---------------------------------------------------------------------//

    std::stringstream rNs;
    rNs<<r.run();
    std::string rN = rNs.str();
    LogDebug("HLTMonBitSummary")<<"this is the beginning of a NEW run: "<< r.run();

    //h1_->Reset();
    // h2_->Reset();
    //pf_->Reset();
    //if (ratio_) ratio_->Reset();

    for (uint i=0; i < nValidTriggers_ && i < 200 ; ++i){
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
  if(configFlag_) return;
     
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

  //cout << " Was at least one path run? " << trh->wasrun() << endl;;
  //cout << " Has at least one path accepted the event? " << trh->accept() << endl;
  //cout << " Has any path encountered an error? " << trh->error() << endl;
  //cout << " Number of paths stored =  " << trh->size() << endl;  

  for (unsigned int trig=0; trig < nValidTriggers_; trig++) {
    //convert *all* trigger names (from config and key) to trigger index properly  
    HLTPathsByIndex_[trig]=triggerNames_.triggerIndex(HLTPathsByName_[trig]);
  }
  
  //get filter information for config triggers only
  for (unsigned int trig=0; trig < nValidConfigTriggers_; trig++) {
    //cout << "Trigger Name = " << HLTPathNamesConfig_[trig] << ", HLTPathsByIndex_ = " << HLTPathsByIndex_[trig] << endl; 
    //cout << "Trigger Name = " << HLTPathNamesConfig_[trig] << ", trh->index = " << lastModule << " " << trh->accept(HLTPathsByIndex_[trig]) << endl; 
    
    //check if trigger exists in TriggerResults
    if(!filterTypes_.empty() && !filterFlag_ && HLTPathsByIndex_[trig] < trh->size()) {
      lastModule = trh->index(HLTPathsByIndex_[trig]);
	
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
void HLTMonBitSummary::beginJob() {  

  if(dbe_){
    if (directory_ != "" && directory_.substr(directory_.length()-1,1) != "/" ) directory_ = directory_+"/" ;
  
    float min = -0.5;
    float max = 200-0.5;
    //uint nBin = HLTPathsByName_.size();
    uint nBin = 200;
   
    dbe_->setCurrentFolder(directory_);

    h1_ = dbe_->book1D("PassingBits_Summary_"+histLabel_,"PassingBits_Summary_"+histLabel_, nBin, min, max);
    h2_ = dbe_->book2D("PassingBits_Correlation_"+histLabel_,"PassingBits_Correlation_"+histLabel_,nBin,min,max, nBin,min,max);
    pf_ = dbe_->book1D("Efficiency_Summary_"+histLabel_,"Efficiency_Summary_"+histLabel_, nBin, min, max);
    if (denominator_!="")
      //ratio_ = dbe_->book1D(std::string("Ratio_"+denominator_),std::string("Ratio_"+denominator_),nBin,min,max);
      ratio_ = dbe_->book1D("HLTRate_"+histLabel_,"HLTRate_"+histLabel_,nBin,min,max);
    else 
      ratio_=0;

  }

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

