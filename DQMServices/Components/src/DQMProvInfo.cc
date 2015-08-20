/*
 * \file DQMProvInfo.cc
 * \author A.Raval / A.Meyer - DESY
 * Last Update:
 *
 */

#include "DQMProvInfo.h"
#include <TSystem.h>
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

const static int XBINS=2500;
const static int YBINS=28;

DQMProvInfo::DQMProvInfo(const edm::ParameterSet& ps){
  
  parameters_ = ps;
  
  dbe_ = edm::Service<DQMStore>().operator->();
  globalTag_           = "MODULE::DEFAULT"; 
  runType_             = parameters_.getUntrackedParameter<std::string>("runType", "No run type selected") ;
  provinfofolder_      = parameters_.getUntrackedParameter<std::string>("provInfoFolder", "ProvInfo") ;
  subsystemname_       = parameters_.getUntrackedParameter<std::string>("subSystemFolder", "Info") ;
  L1gt_                = consumes<L1GlobalTriggerReadoutRecord>(parameters_.getUntrackedParameter<std::string>("L1gt","gtDigis"));
  L1gtEvm_             = consumes<L1GlobalTriggerEvmReadoutRecord>(parameters_.getUntrackedParameter<std::string>("L1gtEvm","gtEvmDigis"));
  dcsStatusCollection_ = consumes<DcsStatusCollection>(parameters_.getUntrackedParameter<std::string>("dcsStatusCollection","scalersRawToDigi"));
  
  // initialize
  nameProcess_ = "HLT"; // the process name is not contained in this ps
  gotProcessParameterSet_=false;
  physDecl_=true; // set true and switch off in case a single event in a given LS does not have it set.
  for (int i=0;i<25;i++) dcs25[i]=true;
  lastlumi_=0;
}

DQMProvInfo::~DQMProvInfo(){
}

void 
DQMProvInfo::beginRun(const edm::Run& r, const edm::EventSetup &c ) {

  makeProvInfo();
  makeHLTKeyInfo(r,c);

  dbe_->cd(); 
  dbe_->setCurrentFolder(subsystemname_ +"/EventInfo/");

  reportSummary_=dbe_->bookFloat("reportSummary");
  reportSummaryMap_ = dbe_->book2D("reportSummaryMap",
                     "HV and Beam Status vs Lumi", XBINS, 1., XBINS+1, YBINS+1, 0., YBINS+1);
  reportSummaryMap_->setBinLabel(1," CSC+",2);   
  reportSummaryMap_->setBinLabel(2," CSC-",2);   
  reportSummaryMap_->setBinLabel(3," DT0",2);    
  reportSummaryMap_->setBinLabel(4," DT+",2);    
  reportSummaryMap_->setBinLabel(5," DT-",2);    
  reportSummaryMap_->setBinLabel(6," EB+",2);    
  reportSummaryMap_->setBinLabel(7," EB-",2);    
  reportSummaryMap_->setBinLabel(8," EE+",2);    
  reportSummaryMap_->setBinLabel(9," EE-",2);    
  reportSummaryMap_->setBinLabel(10,"ES+",2);    
  reportSummaryMap_->setBinLabel(11,"ES-",2);   
  reportSummaryMap_->setBinLabel(12,"HBHEa",2); 
  reportSummaryMap_->setBinLabel(13,"HBHEb",2); 
  reportSummaryMap_->setBinLabel(14,"HBHEc",2); 
  reportSummaryMap_->setBinLabel(15,"HF",2);    
  reportSummaryMap_->setBinLabel(16,"HO",2);    
  reportSummaryMap_->setBinLabel(17,"BPIX",2);  
  reportSummaryMap_->setBinLabel(18,"FPIX",2);  
  reportSummaryMap_->setBinLabel(19,"RPC",2);   
  reportSummaryMap_->setBinLabel(20,"TIBTID",2);
  reportSummaryMap_->setBinLabel(21,"TOB",2);   
  reportSummaryMap_->setBinLabel(22,"TECp",2);  
  reportSummaryMap_->setBinLabel(23,"TECm",2);  
  reportSummaryMap_->setBinLabel(24,"CASTOR",2);
  reportSummaryMap_->setBinLabel(25,"ZDC",2);
  reportSummaryMap_->setBinLabel(26,"PhysDecl",2);
  reportSummaryMap_->setBinLabel(27,"13 TeV",2);
  reportSummaryMap_->setBinLabel(28,"Stable B",2);
  reportSummaryMap_->setBinLabel(29,"Valid",2);
  reportSummaryMap_->setAxisTitle("Luminosity Section");
  reportSummaryMap_->getTH2F()->SetBit(TH1::kCanRebin);

  dbe_->cd();  
  dbe_->setCurrentFolder(subsystemname_ +"/LhcInfo/");
  hBeamMode_=dbe_->book1D("beamMode","beamMode",XBINS,1.,XBINS+1);
  hBeamMode_->getTH1F()->GetYaxis()->Set(21,0.5,21.5);
  hBeamMode_->getTH1F()->SetMaximum(21.5);
  hBeamMode_->getTH1F()->SetBit(TH1::kCanRebin);

  hBeamMode_->setAxisTitle("Luminosity Section",1);
  hBeamMode_->setBinLabel(1,"no mode",2);
  hBeamMode_->setBinLabel(2,"setup",2);
  hBeamMode_->setBinLabel(3,"inj pilot",2);
  hBeamMode_->setBinLabel(4,"inj intr",2);
  hBeamMode_->setBinLabel(5,"inj nomn",2);
  hBeamMode_->setBinLabel(6,"pre ramp",2);
  hBeamMode_->setBinLabel(7,"ramp",2);
  hBeamMode_->setBinLabel(8,"flat top",2);
  hBeamMode_->setBinLabel(9,"squeeze",2);
  hBeamMode_->setBinLabel(10,"adjust",2);
  hBeamMode_->setBinLabel(11,"stable",2);
  hBeamMode_->setBinLabel(12,"unstable",2);
  hBeamMode_->setBinLabel(13,"beam dump",2);
  hBeamMode_->setBinLabel(14,"ramp down",2);
  hBeamMode_->setBinLabel(15,"recovery",2);
  hBeamMode_->setBinLabel(16,"inj dump",2);
  hBeamMode_->setBinLabel(17,"circ dump",2);
  hBeamMode_->setBinLabel(18,"abort",2);
  hBeamMode_->setBinLabel(19,"cycling",2);
  hBeamMode_->setBinLabel(20,"warn b-dump",2);
  hBeamMode_->setBinLabel(21,"no beam",2);
  hBeamMode_->setBinContent(0.,22.);
  

  hLhcFill_=dbe_->book1D("lhcFill","LHC Fill Number",XBINS,1.,XBINS+1);
  hLhcFill_->setAxisTitle("Luminosity Section",1);
  hLhcFill_->getTH1F()->SetBit(TH1::kCanRebin);
  
  hMomentum_=dbe_->book1D("momentum","Beam Energy [GeV]",XBINS,1.,XBINS+1);
  hMomentum_->setAxisTitle("Luminosity Section",1);
  hMomentum_->getTH1F()->SetBit(TH1::kCanRebin);

  hIntensity1_=dbe_->book1D("intensity1","Intensity Beam 1",XBINS,1.,XBINS+1);
  hIntensity1_->setAxisTitle("Luminosity Section",1);
  hIntensity1_->setAxisTitle("N [E10]",2);
  hIntensity1_->getTH1F()->SetBit(TH1::kCanRebin);
  hIntensity2_=dbe_->book1D("intensity2","Intensity Beam 2",XBINS,1.,XBINS+1);
  hIntensity2_->setAxisTitle("Luminosity Section",1);
  hIntensity2_->setAxisTitle("N [E10]",2);
  hIntensity2_->getTH1F()->SetBit(TH1::kCanRebin);

  dbe_->cd();  
  dbe_->setCurrentFolder(subsystemname_ +"/ProvInfo/");
  hIsCollisionsRun_ = dbe_->bookInt("isCollisionsRun");
  hIsCollisionsRun_->Fill(0);
  
  // initialize
  physDecl_=true;
  for (int i=0;i<25;i++) dcs25[i]=true;
  lastlumi_=0;
} 

void DQMProvInfo::analyze(const edm::Event& e, const edm::EventSetup& c){
  if(!gotProcessParameterSet_){
    gotProcessParameterSet_=true;
    edm::ParameterSet ps;
    //fetch the real process name
    nameProcess_ = e.processHistory()[e.processHistory().size()-1].processName();
    e.getProcessParameterSet(nameProcess_,ps);
    globalTag_ = ps.getParameterSet("PoolDBESSource@GlobalTag").getParameter<std::string>("globaltag");
    versGlobaltag_->Fill(globalTag_);
  }
  
  makeDcsInfo(e);
  makeGtInfo(e);

  return;
}

void
DQMProvInfo::endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c)
{

  int nlumi = l.id().luminosityBlock();
  
  edm::LogInfo("DQMProvInfo") << "nlumi: " <<  nlumi << " / number of bins: " << hBeamMode_->getTH1F()->GetNbinsX() << std::endl;

  if (nlumi <= lastlumi_ ) return;

  hBeamMode_->setBinContent(nlumi,beamMode_);
  hLhcFill_->setBinContent(nlumi,lhcFill_);
  hMomentum_->setBinContent(nlumi,momentum_);
  hIntensity1_->setBinContent(nlumi,intensity1_);
  hIntensity2_->setBinContent(nlumi,intensity2_);

  // set to previous in case there was a jump or no previous fill
  for (int l=lastlumi_+1;l<nlumi;l++)
  {
    // setting valid flag to zero for missed LSs
    reportSummaryMap_->setBinContent(l,YBINS+1,0.);
    // setting all other bins to -1 for missed LSs
    for (int i=0;i<YBINS;i++)
      reportSummaryMap_->setBinContent(l,i+1,-1.);
  }

      
  // fill dcs vs lumi
  reportSummaryMap_->setBinContent(nlumi,YBINS+1,1.);
  for (int i=0;i<25;i++)
  {
    if (dcs25[i])
      reportSummaryMap_->setBinContent(nlumi,i+1,1.);
    else
      reportSummaryMap_->setBinContent(nlumi,i+1,0.);

    // set next lumi to -1 for better visibility
    if (nlumi < XBINS)
      reportSummaryMap_->setBinContent(nlumi+1,i+1,-1.);
    dcs25[i]=true;
  }

  // fill physics decl. bit in y bin 26
  if (physDecl_) 
  {
    reportSummary_->Fill(1.); 
    reportSummaryMap_->setBinContent(nlumi,25+1,1.);
    if (nlumi < XBINS) 
      reportSummaryMap_->setBinContent(nlumi+1,25+1,-1.);
  }
  else
  {
    reportSummary_->Fill(0.); 
    reportSummaryMap_->setBinContent(nlumi,25+1,0.);
    if (nlumi < XBINS) 
      reportSummaryMap_->setBinContent(nlumi+1,25+1,-1.);
  }
  
  // fill 7 TeV bit in y bin 27
  if (momentum_ == 3500 || momentum_ == 4000 ) 
  {
    reportSummary_->Fill(1.); 
    reportSummaryMap_->setBinContent(nlumi,26+1,1.);
    if (nlumi < XBINS) 
      reportSummaryMap_->setBinContent(nlumi+1,26+1,-1.);
  }
  else
  {
    reportSummary_->Fill(0.); 
    reportSummaryMap_->setBinContent(nlumi,26+1,0.);
    if (nlumi < XBINS) 
      reportSummaryMap_->setBinContent(nlumi+1,26+1,-1.);
  }

  // fill stable beams bit in y bin 28
  if (beamMode_ == 11) 
  {
    hIsCollisionsRun_->Fill(1);
    reportSummary_->Fill(1.); 
    reportSummaryMap_->setBinContent(nlumi,27+1,1.);
    if (nlumi < XBINS) 
      reportSummaryMap_->setBinContent(nlumi+1,27+1,-1.);
  }
  else
  {
    reportSummary_->Fill(0.); 
    reportSummaryMap_->setBinContent(nlumi,27+1,0.);
    if (nlumi < XBINS) 
      reportSummaryMap_->setBinContent(nlumi+1,27+1,-1.);
  }

  // reset   
  physDecl_=true;  
  lastlumi_=nlumi;

  return;
  
}

void 
DQMProvInfo::makeProvInfo()
{
    dbe_->cd() ;
    dbe_->setCurrentFolder( subsystemname_ + "/" +  provinfofolder_) ;

    // if (dbe_->get("ProvInfo/CMSSW")) return ;
    
    versCMSSW_     = dbe_->bookString("CMSSW",edm::getReleaseVersion().c_str() );
    hostName_      = dbe_->bookString("hostName",gSystem->HostName());
    workingDir_    = dbe_->bookString("workingDir",gSystem->pwd());
    processId_     = dbe_->bookInt("processID"); processId_->Fill(gSystem->GetPid());

    //versDataset_   = dbe_->bookString("Dataset",workflow_);
    versGlobaltag_ = dbe_->bookString("Globaltag",globalTag_);
    versRuntype_ = dbe_->bookString("Run Type",runType_);

    isComplete_ = dbe_->bookInt("runIsComplete"); 
    //isComplete_->Fill((runIsComplete_?1:0));
    fileVersion_ = dbe_->bookInt("fileVersion");
    //fileVersion_->Fill(version_);
    
    return ;
}
void 
DQMProvInfo::makeDcsInfo(const edm::Event& e)
{

  edm::Handle<DcsStatusCollection> dcsStatus;
  e.getByToken(dcsStatusCollection_, dcsStatus);
  for (DcsStatusCollection::const_iterator dcsStatusItr = dcsStatus->begin(); 
                            dcsStatusItr != dcsStatus->end(); ++dcsStatusItr) 
  {
      // edm::LogInfo("DQMProvInfo") << "DCS status: 0x" << std::hex << dcsStatusItr->ready() << std::dec << std::endl;
      if (!dcsStatusItr->ready(DcsStatus::CSCp))   dcs25[0]=false;
      if (!dcsStatusItr->ready(DcsStatus::CSCm))   dcs25[1]=false;   
      if (!dcsStatusItr->ready(DcsStatus::DT0))    dcs25[2]=false;
      if (!dcsStatusItr->ready(DcsStatus::DTp))    dcs25[3]=false;
      if (!dcsStatusItr->ready(DcsStatus::DTm))    dcs25[4]=false;
      if (!dcsStatusItr->ready(DcsStatus::EBp))    dcs25[5]=false;
      if (!dcsStatusItr->ready(DcsStatus::EBm))    dcs25[6]=false;
      if (!dcsStatusItr->ready(DcsStatus::EEp))    dcs25[7]=false;
      if (!dcsStatusItr->ready(DcsStatus::EEm))    dcs25[8]=false;
      if (!dcsStatusItr->ready(DcsStatus::ESp))    dcs25[9]=false;
      if (!dcsStatusItr->ready(DcsStatus::ESm))    dcs25[10]=false; 
      if (!dcsStatusItr->ready(DcsStatus::HBHEa))  dcs25[11]=false;
      if (!dcsStatusItr->ready(DcsStatus::HBHEb))  dcs25[12]=false;
      if (!dcsStatusItr->ready(DcsStatus::HBHEc))  dcs25[13]=false; 
      if (!dcsStatusItr->ready(DcsStatus::HF))     dcs25[14]=false;
      if (!dcsStatusItr->ready(DcsStatus::HO))     dcs25[15]=false;
      if (!dcsStatusItr->ready(DcsStatus::BPIX))   dcs25[16]=false;
      if (!dcsStatusItr->ready(DcsStatus::FPIX))   dcs25[17]=false;
      if (!dcsStatusItr->ready(DcsStatus::RPC))    dcs25[18]=false;
      if (!dcsStatusItr->ready(DcsStatus::TIBTID)) dcs25[19]=false;
      if (!dcsStatusItr->ready(DcsStatus::TOB))    dcs25[20]=false;
      if (!dcsStatusItr->ready(DcsStatus::TECp))   dcs25[21]=false;
      if (!dcsStatusItr->ready(DcsStatus::TECm))   dcs25[22]=false;
      if (!dcsStatusItr->ready(DcsStatus::CASTOR)) dcs25[23]=false;
      if (!dcsStatusItr->ready(DcsStatus::ZDC))    dcs25[24]=false;
  }
      
  return ;
}
void 
DQMProvInfo::makeHLTKeyInfo(const edm::Run& r, const edm::EventSetup &c ) 
{
  
  std::string hltKey = "";
  HLTConfigProvider hltConfig;
  bool changed( true );
  if ( ! hltConfig.init( r, c, nameProcess_, changed) ) 
  {
    // edm::LogInfo("DQMProvInfo") << "errorHltConfigExtraction" << std::endl;
    hltKey = "error extraction" ;
  } 
  else if ( hltConfig.size() <= 0 ) 
  {
   // edm::LogInfo("DQMProvInfo") << "hltConfig" << std::endl;
    hltKey = "error key of length 0" ;
  } 
  else 
  {
    edm::LogInfo("DQMProvInfo") << "HLT key (run)  : " << hltConfig.tableName() << std::endl;
    hltKey =  hltConfig.tableName() ;
  }

  dbe_->cd() ;
  dbe_->setCurrentFolder( subsystemname_ + "/" +  provinfofolder_) ;
  hHltKey_= dbe_->bookString("hltKey",hltKey);

  return ;
  
}
void 
DQMProvInfo::makeGtInfo(const edm::Event& e)
{

  edm::Handle<L1GlobalTriggerReadoutRecord> gtrr_handle;
  e.getByToken(L1gt_, gtrr_handle);
  L1GlobalTriggerReadoutRecord const* gtrr = gtrr_handle.product();
  L1GtFdlWord fdlWord ; 
  if (gtrr)
    fdlWord = gtrr->gtFdlWord();
  else
  {
    edm::LogWarning("DQMProvInfo") << " phys decl. bit not accessible !!!" ;
    physDecl_=false;
    return;
  }
  // cout << "phys decl. bit =" << static_cast<int>(fdlWord.physicsDeclared()) << endl;
  if (fdlWord.physicsDeclared() !=1) physDecl_=false;


  //
  edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvm_handle;
  e.getByToken(L1gtEvm_, gtEvm_handle);
  L1GlobalTriggerEvmReadoutRecord const* gtevm = gtEvm_handle.product();

  L1GtfeWord gtfeEvmWord;
  L1GtfeExtWord gtfeEvmExtWord;
  if (gtevm)
  {
     gtfeEvmWord = gtevm->gtfeWord();
     gtfeEvmExtWord = gtevm->gtfeWord();
  }
  else
    edm::LogInfo("DQMProvInfo") << " gtfeEvmWord inaccessible" ;
   
  lhcFill_ = gtfeEvmExtWord.lhcFillNumber();
  beamMode_ = gtfeEvmExtWord.beamMode();
  momentum_ = gtfeEvmExtWord.beamMomentum();
  intensity1_ = gtfeEvmExtWord.totalIntensityBeam1();
  intensity2_ = gtfeEvmExtWord.totalIntensityBeam2();
  
  edm::LogInfo("DQMProvInfo") << lhcFill_ << " " << beamMode_ << " " 
            << momentum_ << " " 
	    << intensity1_ << " " << intensity2_ 
	    << std::endl;

  return;
}
