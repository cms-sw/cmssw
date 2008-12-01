/*  \author Anna Cimmino*/
#include <DQM/RPCMonitorClient/interface/RPCFEDIntegrity.h>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//DQM Services
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//EventFilter
#include "EventFilter/RPCRawToDigi/interface/RPCRawDataCounts.h"
#include "EventFilter/RPCRawToDigi/interface/DataRecord.h"

using namespace edm;
using namespace std;
RPCFEDIntegrity::RPCFEDIntegrity(const ParameterSet& ps ) {
  LogVerbatim ("rpceventsummary") << "[RPCFEDIntegrity]: Constructor";

  prefixDir_ = ps.getUntrackedParameter<string>("RPCPrefixDir", "RPC");
  merge_ = ps.getUntrackedParameter<bool>("MergeRuns", false);
  numOfFED_ =  ps.getUntrackedParameter<int>("NumberOfFED", 3);
  minFEDNum_ =  ps.getUntrackedParameter<int>("MinimumFEDID", 790);
  maxFEDNum_ =  ps.getUntrackedParameter<int>("MaximumFEDID", 792);

  histoName_.push_back("FEDEntries");
  histoName_.push_back("FEDFatal");
  histoName_.push_back("FEDNonFatal");
  
  FATAL_LIMIT = 5;

  init_ = false;
}

RPCFEDIntegrity::~RPCFEDIntegrity(){
  LogVerbatim ("rpceventsummary") << "[RPCFEDIntegrity]: Destructor ";
  //  dbe_=0;
}

void RPCFEDIntegrity::beginJob(const EventSetup& iSetup){
 LogVerbatim ("rpceventsummary") << "[RPCFEDIntegrity]: Begin job ";
 dbe_ = Service<DQMStore>().operator->();
}

void RPCFEDIntegrity::beginRun(const Run& r, const EventSetup& c){
 LogVerbatim ("rpceventsummary") << "[RPCFEDIntegrity]: Begin run ";

 if (!init_) this->bookFEDMe();
 else if (!merge_) this->reset();
}

void RPCFEDIntegrity::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCFEDIntegrity::analyze(const Event& iEvent, const EventSetup& c) {

  //get hold of raw data counts
  Handle<RPCRawDataCounts> rawCounts;
  iEvent.getByType( rawCounts);

  const RPCRawDataCounts * aCounts = rawCounts.product();
  //  const RPCRawDataCounts * theCounts += *aCounts;

  vector<double> v1;
  map<int,double> fedOccupancy;

  MonitorElement * me;

  //loop  on all FEDS
  for (int fedId=minFEDNum_ ;fedId<maxFEDNum_+1;fedId++) {
    v1.clear(); // v1 is cleared in  recordTypeVector() but you never know
    aCounts->recordTypeVector(fedId,v1); 
    
    bool fatal = false;
    bool nonfatal = false;
    unsigned int err = 2; // err = 0,1 means non problems,we 

    //loop on errors
    while( err<(v1.size()-1) && (!fatal || !nonfatal)){
      if(v1[err]<= FATAL_LIMIT && v1[err+1]!=0) { 
	fatal=true;
	break;
      } else if (v1[err+1]!=0) {
	nonfatal = true;
	break;
      }
      err ++;
    }//end loop o errors
    
    me = dbe_->get(prefixDir_+"/FEDIntegrity/FEDEntries");
    if(me!=0 && v1.size()!=0) me->Fill(fedId);

    me= dbe_->get(prefixDir_+"/FEDIntegrity/FEDFatal");
    if(me!=0 && v1.size()!=0) me->Fill(fedId);

  }//end loop on all FEDs
}

void RPCFEDIntegrity::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {}

void RPCFEDIntegrity::endRun(const Run& r, const EventSetup& c){}

void RPCFEDIntegrity::endJob(){
  dbe_=0;
}

//Fill report summary
void  RPCFEDIntegrity::bookFEDMe(void){

  if(dbe_){
    dbe_->setCurrentFolder(prefixDir_+"/FEDIntegrity/");
   
    for(unsigned int i = 0; i<histoName_.size(); i++){
     dbe_->book1D(histoName_[i].c_str(),histoName_[i].c_str(),numOfFED_, minFEDNum_, maxFEDNum_ +1);
    }
  }

  init_ = true;
}

void  RPCFEDIntegrity::reset(void){

  MonitorElement * me;  

  if(dbe_){
    for(unsigned int i = 0; i<histoName_.size(); i++){
      if( me = dbe_->get(prefixDir_ +"FEDIntegrity/"+ histoName_[i]) ) 	me->Reset();
    }
  }
}
