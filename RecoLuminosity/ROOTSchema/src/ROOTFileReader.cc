#include "RecoLuminosity/ROOTSchema/interface/ROOTFileReader.h"
#include <TROOT.h>
#include <TFile.h>

#include <iomanip>
#include <sstream>

HCAL_HLX::ROOTFileReader::ROOTFileReader(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
  
  mFileName_ = "";

  mChain_ = new TChain("LumiTree");

  lumiSection_ = new HCAL_HLX::LUMI_SECTION;
  Threshold_       = new HCAL_HLX::LUMI_THRESHOLD;
  L1Trigger_       = new HCAL_HLX::LEVEL1_TRIGGER;
  HLT_             = new HCAL_HLX::HLT;
  TriggerDeadtime_ = new HCAL_HLX::TRIGGER_DEADTIME;
  RingSet_         = new HCAL_HLX::LUMI_HF_RING_SET;


#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

HCAL_HLX::ROOTFileReader::~ROOTFileReader(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
  
  delete mChain_;
  
  delete Threshold_;
  delete L1Trigger_;
  delete HLT_;
  delete TriggerDeadtime_;
  delete RingSet_;

  delete lumiSection_;

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

int HCAL_HLX::ROOTFileReader::ReplaceFile(const std::string& fileName){
  // replacing a file changes the run number and section number automatically.

#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
  
  std::stringstream branchName;
  
  delete mChain_;

  mChain_ = new TChain("LumiTree");

  mChain_->Add(fileName.c_str());
  

  // HCAL_HLX::LUMI_SECTION 
  
  Header_  = &(lumiSection_->hdr);
  Summary_ = &(lumiSection_->lumiSummary);
  Detail_  = &(lumiSection_->lumiDetail);

  mChain_->SetBranchAddress("Header.",  &Header_,  &b_Header);
  mChain_->SetBranchAddress("Summary.", &Summary_, &b_Summary);
  mChain_->SetBranchAddress("Detail.",  &Detail_,  &b_Detail);

  for(int HLXnum = 0; HLXnum < HCAL_HLX_MAX_HLXS; HLXnum++){
    EtSumPtr[HLXnum] = &(lumiSection_->etSum[HLXnum]);
    branchName.str(std::string());
    branchName << "ETSum" << std::setw(2) << std::setfill('0') << HLXnum << ".";
    mChain_->SetBranchAddress(branchName.str().c_str(), &EtSumPtr[HLXnum], &b_ETSum[HLXnum]);

    OccupancyPtr[HLXnum] = &(lumiSection_->occupancy[HLXnum]);
    branchName.str(std::string());
    branchName << "Occupancy" << std::setw(2) << std::setfill('0') << HLXnum << ".";
    mChain_->SetBranchAddress(branchName.str().c_str(), &OccupancyPtr[HLXnum], &b_Occupancy[HLXnum]);

    LHCPtr[HLXnum] = &(lumiSection_->lhc[HLXnum]);
    branchName.str(std::string());
    branchName << "LHC" << std::setw(2) << std::setfill('0') << HLXnum << ".";
    mChain_->SetBranchAddress(branchName.str().c_str(), &LHCPtr[HLXnum], &b_LHC[HLXnum]);
  }

  // OTHER
  mChain_->SetBranchAddress("Threshold.",        &Threshold_,       &b_Threshold);
  mChain_->SetBranchAddress("Level1_Trigger.",   &L1Trigger_,       &b_L1Trigger);
  mChain_->SetBranchAddress("HLT.",              &HLT_,             &b_HLT);
  mChain_->SetBranchAddress("Trigger_Deadtime.", &TriggerDeadtime_, &b_TriggerDeadtime);
  mChain_->SetBranchAddress("HF_Ring_Set.",      &RingSet_,         &b_RingSet);
  
  // Get run and section number.
  mChain_->GetEntry(0);
    
  runNumber_     = lumiSection_->hdr.runNumber;
  sectionNumber_ = lumiSection_->hdr.sectionNumber;

#ifdef DEBUG
  std::cout << "***** Run Number: " << runNumber_ << " *****" << std::endl;
  std::cout << "***** Section Number: " << sectionNumber_ << " *****" << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif

  return 0;
}

int HCAL_HLX::ROOTFileReader::GetEntry(int entry){

  return mChain_->GetEntry(entry);
}

int HCAL_HLX::ROOTFileReader::GetNumEntries(){

  return mChain_->GetEntries();
}

int HCAL_HLX::ROOTFileReader::GetLumiSection(HCAL_HLX::LUMI_SECTION& localSection){

  memcpy(&localSection, lumiSection_, sizeof(HCAL_HLX::LUMI_SECTION));

  return 0;
}

int HCAL_HLX::ROOTFileReader::GetThreshold(HCAL_HLX::LUMI_THRESHOLD&  localThreshold){

  memcpy(&localThreshold, Threshold_, sizeof(localThreshold));
  return 0;
}

int HCAL_HLX::ROOTFileReader::GetHFRingSet(HCAL_HLX::LUMI_HF_RING_SET& localRingSet){

  memcpy(&localRingSet, RingSet_, sizeof(localRingSet));
  return 0;
}


int HCAL_HLX::ROOTFileReader::GetL1Trigger(HCAL_HLX::LEVEL1_TRIGGER& localL1Trigger){

  memcpy(&localL1Trigger, L1Trigger_, sizeof(localL1Trigger));
  return 0;
}

int HCAL_HLX::ROOTFileReader::GetHLT(HCAL_HLX::HLT& localHLT){

  memcpy(&localHLT, HLT_, sizeof(localHLT));
  return 0;
}

int HCAL_HLX::ROOTFileReader::GetTriggerDeadtime(HCAL_HLX::TRIGGER_DEADTIME& localTD){

  memcpy(&localTD, TriggerDeadtime_, sizeof(localTD));
  return 0;
}
