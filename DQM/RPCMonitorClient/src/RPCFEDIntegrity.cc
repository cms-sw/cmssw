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
    v1.clear();
    aCounts->recordTypeVector(fedId,v1); 

    if(fedOccupancy.find(fedId)== fedOccupancy.end() || fedOccupancy.size()==0) fedOccupancy[fedId]=0;
    
    //loop on errors
    for (unsigned int err = 1 ; err<v1.size(); err +=2){//get onlz even elements of the vector
       fedOccupancy[fedId] += v1[err];

      if(err-1!=0 && err-1 <= FATAL_LIMIT){
	me= dbe_->get(prefixDir_+"/FEDIntegrity/FEDFatal");
	me ->Fill(fedId,v1[err]);
      }
      else if (err-1!=0){
	me= dbe_->get(prefixDir_+"/FEDIntegrity/FEDNonFatal");
	me ->Fill(fedId,v1[err]);
      }

    }//end loop o errors

      me = dbe_->get(prefixDir_+"/FEDIntegrity/FEDEntries");

      if(me!=0) me->Fill(fedId, fedOccupancy[fedId] );
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
