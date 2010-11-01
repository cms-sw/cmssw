/*
 * \file DQMProvInfo.cc
 * \author A.Raval / A.Meyer - DESY
 * Last Update:
 * $Date: 2010/05/20 00:30:52 $
 * $Revision: 1.22 $
 * $Author: ameyer $
 *
 */

#include "DQMProvInfo.h"
#include <TSystem.h>
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

const static int XBINS=2500;
const static int YBINS=27;

DQMProvInfo::DQMProvInfo(const edm::ParameterSet& ps){
  
  parameters_ = ps;

  dbe_ = edm::Service<DQMStore>().operator->();

  provinfofolder_ = parameters_.getUntrackedParameter<std::string>("provInfoFolder", "ProvInfo") ;
  subsystemname_ = parameters_.getUntrackedParameter<std::string>("subSystemFolder", "Info") ;
  
  // initialize
  physDecl_=true; // set true and switch off in case a single event in a given LS does not have it set.
  for (int i=0;i<24;i++) dcs24[i]=true;
  lastlumi_=0;
}

DQMProvInfo::~DQMProvInfo(){
}

void 
DQMProvInfo::beginRun(const edm::Run& r, const edm::EventSetup &c ) {

  makeProvInfo();

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
  reportSummaryMap_->setBinLabel(25,"PhysDecl",2);
  reportSummaryMap_->setBinLabel(26,"7 TeV",2);
  reportSummaryMap_->setBinLabel(27,"Stable B",2);
  reportSummaryMap_->setBinLabel(28,"Valid",2);
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

  // initialize
  physDecl_=true;
  for (int i=0;i<24;i++) dcs24[i]=true;
  lastlumi_=0;
} 

void DQMProvInfo::analyze(const edm::Event& e, const edm::EventSetup& c){
 
  makeDcsInfo(e);
  makeGtInfo(e);

  return;
}

void
DQMProvInfo::endLuminosityBlock(const edm::LuminosityBlock& l, const edm::EventSetup& c)
{

  int nlumi = l.id().luminosityBlock();
  
  std::cout << "nlumi: " <<  nlumi << " / number of bins: " << hBeamMode_->getTH1F()->GetNbinsX() << std::endl;

  if (nlumi <= lastlumi_ ) return;

  hBeamMode_->setBinContent(nlumi,beamMode_);
  hLhcFill_->setBinContent(nlumi,lhcFill_);
  hMomentum_->setBinContent(nlumi,momentum_);
  hIntensity1_->setBinContent(nlumi,intensity1_);
  hIntensity2_->setBinContent(nlumi,intensity2_);

  // set to previous in case there was a jump or no previous fill
  for (int l=lastlumi_+1;l<nlumi;l++)
  {
    if (lastlumi_ > 0 && reportSummaryMap_->getBinContent(lastlumi_,YBINS+1) == 1) 
    {
      reportSummaryMap_->setBinContent(l,YBINS+1,0.);
      for (int i=0;i<YBINS;i++)
      {
	  float lastvalue = reportSummaryMap_->getBinContent(lastlumi_,i+1);
	  reportSummaryMap_->setBinContent(l,i+1,lastvalue);
      }
    }
    else
    {
      reportSummaryMap_->setBinContent(l,YBINS+1,0.);
      for (int i=0;i<YBINS;i++)
	reportSummaryMap_->setBinContent(l,i+1,-1.);
    }
  }

      
  // fill dcs vs lumi
  reportSummaryMap_->setBinContent(nlumi,YBINS+1,1.);
  for (int i=0;i<24;i++)
  {
    if (dcs24[i])
      reportSummaryMap_->setBinContent(nlumi,i+1,1.);
    else
      reportSummaryMap_->setBinContent(nlumi,i+1,0.);

    // set next lumi to -1 for better visibility
    if (nlumi < XBINS)
      reportSummaryMap_->setBinContent(nlumi+1,i+1,-1.);
    dcs24[i]=true;
  }

  // fill physics decl. bit in y bin 25
  if (physDecl_) 
  {
    reportSummary_->Fill(1.); 
    reportSummaryMap_->setBinContent(nlumi,24+1,1.);
    if (nlumi < XBINS) 
      reportSummaryMap_->setBinContent(nlumi+1,24+1,-1.);
  }
  else
  {
    reportSummary_->Fill(0.); 
    reportSummaryMap_->setBinContent(nlumi,24+1,0.);
    if (nlumi < XBINS) 
      reportSummaryMap_->setBinContent(nlumi+1,24+1,-1.);
  }
  
  // fill 7 TeV bit in y bin 26
  if (momentum_ == 3500) 
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

  // fill stable beams bit in y bin 27
  if (beamMode_ == 11) 
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

  // reset   
  physDecl_=true;  
  lastlumi_=nlumi;

  return;
  
}

// run showtag command line
std::string 
DQMProvInfo::getShowTags(void)
{
   TString out;
   FILE *pipe = gSystem->OpenPipe("showtags u -t", "r");

   TString line;
   while (line.Gets(pipe,true)) {
     if (line.Contains("Test Release")) continue;
     if (line.Contains("Base Release")) continue;
     if (line.Contains("Test release")) continue;
     if (line.Contains("--- Tag ---")) continue;
     if (line.Contains(" ")) line.Replace(line.First(" "),1,":");
     line.ReplaceAll(" ","");
     out = out + line + ";";
     if (line.Contains("-------------------")) break;
     if (out.Length()>XBINS) break;
   }
   out.ReplaceAll("--","");
   out.ReplaceAll(";-",";");
   out.ReplaceAll(";;",";");
   out.ReplaceAll("\n","");

   Int_t r = gSystem->ClosePipe(pipe);
   if (r) {
     gSystem->Error("ShowTags","problem running command showtags -u -t");
   }

   std::string str(out);
   if (str.length()>2000) str.resize(2000);

   std::string safestr =
     "/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-;:";
   size_t found=str.find_first_not_of(safestr);
   if (found!=std::string::npos)
   {
     edm::LogWarning("DQMProvInfo::ShowTags") << " Illegal character found: " 
               << str[found] 
               << " at position " 
               << int(found) ;
     return "notags";
   }   
   return str;
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
    versGlobaltag_ = dbe_->bookString("Globaltag","global tag"); // FIXME
    versTaglist_   = dbe_->bookString("Taglist",getShowTags()); 

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
  e.getByLabel("scalersRawToDigi", dcsStatus);
  for (DcsStatusCollection::const_iterator dcsStatusItr = dcsStatus->begin(); 
                            dcsStatusItr != dcsStatus->end(); ++dcsStatusItr) 
  {
      // std::cout << "DCS status: 0x" << std::hex << dcsStatusItr->ready() << std::dec << std::endl;
      if (!dcsStatusItr->ready(DcsStatus::CSCp))   dcs24[0]=false;
      if (!dcsStatusItr->ready(DcsStatus::CSCm))   dcs24[1]=false;   
      if (!dcsStatusItr->ready(DcsStatus::DT0))    dcs24[2]=false;
      if (!dcsStatusItr->ready(DcsStatus::DTp))    dcs24[3]=false;
      if (!dcsStatusItr->ready(DcsStatus::DTm))    dcs24[4]=false;
      if (!dcsStatusItr->ready(DcsStatus::EBp))    dcs24[5]=false;
      if (!dcsStatusItr->ready(DcsStatus::EBm))    dcs24[6]=false;
      if (!dcsStatusItr->ready(DcsStatus::EEp))    dcs24[7]=false;
      if (!dcsStatusItr->ready(DcsStatus::EEm))    dcs24[8]=false;
      if (!dcsStatusItr->ready(DcsStatus::ESp))    dcs24[9]=false;
      if (!dcsStatusItr->ready(DcsStatus::ESm))    dcs24[10]=false; 
      if (!dcsStatusItr->ready(DcsStatus::HBHEa))  dcs24[11]=false;
      if (!dcsStatusItr->ready(DcsStatus::HBHEb))  dcs24[12]=false;
      if (!dcsStatusItr->ready(DcsStatus::HBHEc))  dcs24[13]=false; 
      if (!dcsStatusItr->ready(DcsStatus::HF))     dcs24[14]=false;
      if (!dcsStatusItr->ready(DcsStatus::HO))     dcs24[15]=false;
      if (!dcsStatusItr->ready(DcsStatus::BPIX))   dcs24[16]=false;
      if (!dcsStatusItr->ready(DcsStatus::FPIX))   dcs24[17]=false;
      if (!dcsStatusItr->ready(DcsStatus::RPC))    dcs24[18]=false;
      if (!dcsStatusItr->ready(DcsStatus::TIBTID)) dcs24[19]=false;
      if (!dcsStatusItr->ready(DcsStatus::TOB))    dcs24[20]=false;
      if (!dcsStatusItr->ready(DcsStatus::TECp))   dcs24[21]=false;
      if (!dcsStatusItr->ready(DcsStatus::TECm))   dcs24[22]=false;
      if (!dcsStatusItr->ready(DcsStatus::CASTOR)) dcs24[23]=false;
  }
      
  return ;
}

void 
DQMProvInfo::makeGtInfo(const edm::Event& e)
{

  edm::Handle<L1GlobalTriggerReadoutRecord> gtrr_handle;
  e.getByLabel("gtDigis", gtrr_handle);
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
  e.getByLabel("gtEvmDigis", gtEvm_handle);
  L1GlobalTriggerEvmReadoutRecord const* gtevm = gtEvm_handle.product();

  L1GtfeWord gtfeEvmWord;
  L1GtfeExtWord gtfeEvmExtWord;
  if (gtevm)
  {
     gtfeEvmWord = gtevm->gtfeWord();
     gtfeEvmExtWord = gtevm->gtfeWord();
  }
  else
    std::cout << " gtfeEvmWord inaccessible" ;
   
  lhcFill_ = gtfeEvmExtWord.lhcFillNumber();
  beamMode_ = gtfeEvmExtWord.beamMode();
  momentum_ = gtfeEvmExtWord.beamMomentum();
  intensity1_ = gtfeEvmExtWord.totalIntensityBeam1();
  intensity2_ = gtfeEvmExtWord.totalIntensityBeam2();
  
  std::cout << lhcFill_ << " " << beamMode_ << " " 
            << momentum_ << " " 
	    << intensity1_ << " " << intensity2_ 
	    << std::endl;

  return;
}
