#include "DQM/RPCMonitorDigi/interface/RPCDcsInfo.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"


RPCDcsInfo::RPCDcsInfo(const edm::ParameterSet& ps){
  
  dbe_ = edm::Service<DQMStore>().operator->();
  subsystemname_ = ps.getUntrackedParameter<std::string>("subSystemFolder", "RPC") ;
  dcsinfofolder_ = ps.getUntrackedParameter<std::string>("dcsInfoFolder", "DCSInfo") ;
  scalersRawToDigiLabel_ = ps.getUntrackedParameter<std::string>("ScalersRawToDigiLabel", "scalersRawToDigi");
  
  // initialize
  dcs = true;
}

RPCDcsInfo::~RPCDcsInfo(){}

void RPCDcsInfo::beginRun(const edm::Run& r, const edm::EventSetup &c ) {

  dbe_->cd();  
  dbe_->setCurrentFolder(subsystemname_ + "/" + dcsinfofolder_);

  currentDCSStatus_  = dbe_->book2D("DCSStatus","RPC DCS Status",1,0.,1.,1,0.,1.);
  
  currentDCSStatus_->setBinContent(1,1, -1);
  
  dcs=true;
} 

void RPCDcsInfo::analyze(const edm::Event& e, const edm::EventSetup& c){
 
  makeDcsInfo(e);

  if(dcs)currentDCSStatus_ ->setBinContent(1,1, 1.);
  else  currentDCSStatus_ ->setBinContent(1,1, 0.);
  
  dcs = true;
  
  return;
}

void RPCDcsInfo::endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c){}


void  RPCDcsInfo::makeDcsInfo(const edm::Event& e) {

  edm::Handle<DcsStatusCollection> dcsStatus;

  if ( ! e.getByLabel("scalersRawToDigi", dcsStatus) ){
    dcs = false;
    return;
  }
  
  if ( ! dcsStatus.isValid() ) 
  {
    edm::LogWarning("RPCDcsInfo") << "scalersRawToDigi not found" ;
    dcs = false; // info not available: set to false
    return;
  }
    

  for (DcsStatusCollection::const_iterator dcsStatusItr = dcsStatus->begin(); 
                            dcsStatusItr != dcsStatus->end(); ++dcsStatusItr)   {

      if (!dcsStatusItr->ready(DcsStatus::RPC)) dcs=false;
      
  }
      
  return ;
}

