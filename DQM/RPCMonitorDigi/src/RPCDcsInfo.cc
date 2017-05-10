#include "DQM/RPCMonitorDigi/interface/RPCDcsInfo.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"


RPCDcsInfo::RPCDcsInfo(const edm::ParameterSet& ps){
  
  
  subsystemname_ = ps.getUntrackedParameter<std::string>("subSystemFolder", "RPC") ;
  dcsinfofolder_ = ps.getUntrackedParameter<std::string>("dcsInfoFolder", "DCSInfo") ;
  scalersRawToDigiLabel_  = consumes<DcsStatusCollection>(ps.getParameter<edm::InputTag>("ScalersRawToDigiLabel"));
  
}

RPCDcsInfo::~RPCDcsInfo(){}

void RPCDcsInfo::bookHistograms(DQMStore::IBooker & ibooker,
                                edm::Run const & /* iRun */,
                                edm::EventSetup const & /* iSetup */) {

  // Fetch GlobalTag information and fill the string/ME.
  ibooker.cd();
  ibooker.setCurrentFolder(subsystemname_ + "/" + dcsinfofolder_);

  DCSbyLS_ = ibooker.book1D("DCSbyLS","DCS",1,0.5,1.5);
  DCSbyLS_->setLumiFlag();

  // initialize
  dcs=true;

}


void RPCDcsInfo::analyze(const edm::Event& e, const edm::EventSetup& c){
 
  makeDcsInfo(e);
  return;
}

void RPCDcsInfo::endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c){

  // int nlumi = l.id().luminosityBlock();

  // fill dcs vs lumi 
  /* set those bins 0 for which bits are ON 
     needed for merge off lumi histograms across files */
  if (dcs)  DCSbyLS_->setBinContent(1,0.);
  else  DCSbyLS_->setBinContent(1,1.);

  dcs = true;
  
  return;
}


void  RPCDcsInfo::makeDcsInfo(const edm::Event& e) {

  edm::Handle<DcsStatusCollection> dcsStatus;

  if ( ! e.getByToken(scalersRawToDigiLabel_, dcsStatus) ){
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

