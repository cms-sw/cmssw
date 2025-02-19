/*
 * \file DQMDcsInfo.cc
 * \author A.Meyer - DESY
 * Last Update:
 * $Date: 2012/08/02 07:59:10 $
 * $Revision: 1.3 $
 * $Author: rovere $
 *
 */

#include "DQMDcsInfo.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

// Framework

const static int XBINS=2000;

DQMDcsInfo::DQMDcsInfo(const edm::ParameterSet& ps)
{
  
  parameters_ = ps;

  dbe_ = edm::Service<DQMStore>().operator->();

  subsystemname_ = parameters_.getUntrackedParameter<std::string>("subSystemFolder", "Info") ;
  dcsinfofolder_ = parameters_.getUntrackedParameter<std::string>("dcsInfoFolder", "DcsInfo") ;
  
  // initialize
  for (int i=0;i<25;i++) dcs[i]=true;
}

DQMDcsInfo::~DQMDcsInfo(){
}

void 
DQMDcsInfo::beginRun(const edm::Run& r, const edm::EventSetup &c ) {

  // Fetch GlobalTag information and fill the string/ME.
  dbe_->cd();  
  dbe_->setCurrentFolder(subsystemname_ +"/CMSSWInfo/");
  const edm::ParameterSet &globalTagPSet = edm::getProcessParameterSet()
					   .getParameterSet("PoolDBESSource@GlobalTag");

  dbe_->bookString("globalTag_Step1", globalTagPSet.getParameter<std::string>("globaltag"));

  dbe_->cd();  
  dbe_->setCurrentFolder(subsystemname_ + "/" + dcsinfofolder_);

  DCSbyLS_=dbe_->book1D("DCSbyLS","DCS",25,0.,25.);
  DCSbyLS_->setLumiFlag();

  // initialize
  for (int i=0;i<25;i++) dcs[i]=true;
} 

void DQMDcsInfo::analyze(const edm::Event& e, const edm::EventSetup& c){
 
  makeDcsInfo(e);
  makeGtInfo(e);

  return;
}

void
DQMDcsInfo::endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c)
{
  // int nlumi = l.id().luminosityBlock();

  // fill dcs vs lumi 
  /* set those bins 0 for which bits are ON 
     needed for merge off lumi histograms across files */
  for (int i=0;i<25;i++)
  {
    if (dcs[i])
      DCSbyLS_->setBinContent(i+1,0.);
    else
      DCSbyLS_->setBinContent(i+1,1.);

    dcs[i]=true;
  }

  return;
}

void 
DQMDcsInfo::makeDcsInfo(const edm::Event& e)
{

  edm::Handle<DcsStatusCollection> dcsStatus;
  if ( ! e.getByLabel("scalersRawToDigi", dcsStatus) )
  {
    for (int i=0;i<24;i++) dcs[i]=false;
    return;
  }
  
  if ( ! dcsStatus.isValid() ) 
  {
    edm::LogWarning("DQMDcsInfo") << "scalersRawToDigi not found" ;
    for (int i=0;i<24;i++) dcs[i]=false; // info not available: set to false
    return;
  }
    
  for (DcsStatusCollection::const_iterator dcsStatusItr = dcsStatus->begin(); 
                            dcsStatusItr != dcsStatus->end(); ++dcsStatusItr) 
  {
      if (!dcsStatusItr->ready(DcsStatus::CSCp))   dcs[0]=false;
      if (!dcsStatusItr->ready(DcsStatus::CSCm))   dcs[1]=false;   
      if (!dcsStatusItr->ready(DcsStatus::DT0))    dcs[2]=false;
      if (!dcsStatusItr->ready(DcsStatus::DTp))    dcs[3]=false;
      if (!dcsStatusItr->ready(DcsStatus::DTm))    dcs[4]=false;
      if (!dcsStatusItr->ready(DcsStatus::EBp))    dcs[5]=false;
      if (!dcsStatusItr->ready(DcsStatus::EBm))    dcs[6]=false;
      if (!dcsStatusItr->ready(DcsStatus::EEp))    dcs[7]=false;
      if (!dcsStatusItr->ready(DcsStatus::EEm))    dcs[8]=false;
      if (!dcsStatusItr->ready(DcsStatus::ESp))    dcs[9]=false;
      if (!dcsStatusItr->ready(DcsStatus::ESm))    dcs[10]=false; 
      if (!dcsStatusItr->ready(DcsStatus::HBHEa))  dcs[11]=false;
      if (!dcsStatusItr->ready(DcsStatus::HBHEb))  dcs[12]=false;
      if (!dcsStatusItr->ready(DcsStatus::HBHEc))  dcs[13]=false; 
      if (!dcsStatusItr->ready(DcsStatus::HF))     dcs[14]=false;
      if (!dcsStatusItr->ready(DcsStatus::HO))     dcs[15]=false;
      if (!dcsStatusItr->ready(DcsStatus::BPIX))   dcs[16]=false;
      if (!dcsStatusItr->ready(DcsStatus::FPIX))   dcs[17]=false;
      if (!dcsStatusItr->ready(DcsStatus::RPC))    dcs[18]=false;
      if (!dcsStatusItr->ready(DcsStatus::TIBTID)) dcs[19]=false;
      if (!dcsStatusItr->ready(DcsStatus::TOB))    dcs[20]=false;
      if (!dcsStatusItr->ready(DcsStatus::TECp))   dcs[21]=false;
      if (!dcsStatusItr->ready(DcsStatus::TECm))   dcs[22]=false;
      if (!dcsStatusItr->ready(DcsStatus::CASTOR)) dcs[23]=false;
  }
      
  return ;
}

void 
DQMDcsInfo::makeGtInfo(const edm::Event& e)
{

  edm::Handle<L1GlobalTriggerReadoutRecord> gtrr_handle;
  if ( ! e.getByLabel("gtDigis", gtrr_handle) ) 
  {
    dcs[24]=false; // info not available: set to false
    return;
  }
  
  if ( ! gtrr_handle.isValid() ) 
  {
    edm::LogWarning("DQMDcsInfo") << " gtDigis not found" ;
    dcs[24]=false; // info not available: set to false
    return;
  }
  
  L1GlobalTriggerReadoutRecord const* gtrr = gtrr_handle.product();
  L1GtFdlWord fdlWord ; 
  if (gtrr)
    fdlWord = gtrr->gtFdlWord();
  else
  {
    edm::LogWarning ("DQMDcsInfo") << " phys decl. bit not accessible !!!";
    dcs[24]=false; // info not available: set to false
    return;
  }

  if (fdlWord.physicsDeclared() !=1) dcs[24]=false;

  return;
}
