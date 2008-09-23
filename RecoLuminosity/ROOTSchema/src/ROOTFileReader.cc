#include "RecoLuminosity/ROOTSchema/interface/ROOTFileReader.h"

// C
#include <cstring> // memset

// STL
#include <iomanip>
#include <sstream>
#include <algorithm>

// Unix
#include <dirent.h> // opendir

// ROOT
#include <TROOT.h>
#include <TFile.h>
#include <TChain.h>

HCAL_HLX::ROOTFileReader::ROOTFileReader(){
  
  mChain_ = new TChain("LumiTree");
  Init();
}

HCAL_HLX::ROOTFileReader::~ROOTFileReader(){

  CleanUp();
  delete mChain_;  
}

int HCAL_HLX::ROOTFileReader::CreateFileNameList(){

  // Look for files that follow the standard naming convention.
  DIR *dp;
  struct dirent *dirp;
  std::string tempFileName;

  std::vector< std::string > fileNames;
  fileNames.clear();

  if( dirName_ == ""){
    return false;
  }
  
  // Check directory existance.
  if( ( dp = opendir( dirName_.c_str() )  ) == NULL ){
    closedir(dp);
    return false;
  }

  while( (dirp = readdir(dp)) != NULL ){
    tempFileName = dirp->d_name;
    if(tempFileName.substr(0,8) == "CMS_LUMI" ){
      fileNames.push_back( dirName_ + tempFileName);
    }
  }
  closedir(dp);

  if( fileNames.size() == 0 ){
    return false;
  }

  sort(fileNames.begin(), fileNames.end());
  return ReplaceFile( fileNames );
}

int HCAL_HLX::ROOTFileReader::SetFileName(const std::string &fileName){  

  std::vector< std::string > tempVecOfStrings;

  tempVecOfStrings.clear();
  tempVecOfStrings.push_back( dirName_ + fileName);

  return ReplaceFile( tempVecOfStrings );
}

int HCAL_HLX::ROOTFileReader::ReplaceFile(const std::vector< std::string> &fileNames){
  // ReplaceFile is called by either SetFileName or CreateFileNameList.
  
  delete mChain_;
  mChain_ = new TChain("LumiTree");
  
  for( std::vector< std::string >::const_iterator VoS = fileNames.begin(); 
       VoS != fileNames.end(); 
       ++VoS){
    mChain_->Add((*VoS).c_str());
  }

  CreateTree();

  return mChain_->GetEntries();
}

void HCAL_HLX::ROOTFileReader::CreateTree(){

  Header_  = &(lumiSection_->hdr);
  Summary_ = &(lumiSection_->lumiSummary);
  Detail_  = &(lumiSection_->lumiDetail);

  mChain_->SetBranchAddress("Header.",  &Header_,  &b_Header);

  if( !bEtSumOnly_ ){
    mChain_->SetBranchAddress("Summary.", &Summary_, &b_Summary);
    mChain_->SetBranchAddress("Detail.",  &Detail_,  &b_Detail);
  }

  for(unsigned int iHLX = 0; iHLX < 36; ++iHLX){
    std::stringstream branchName;

    EtSumPtr_[iHLX] = &(lumiSection_->etSum[iHLX]);
    branchName.str(std::string());
    branchName << "ETSum" << std::setw(2) << std::setfill('0') << iHLX << ".";
    mChain_->SetBranchAddress(branchName.str().c_str(), &EtSumPtr_[iHLX], &b_ETSum[iHLX]);

    if( !bEtSumOnly_ ){
      OccupancyPtr_[iHLX] = &(lumiSection_->occupancy[iHLX]);
      branchName.str(std::string());
      branchName << "Occupancy" << std::setw(2) << std::setfill('0') << iHLX << ".";
      mChain_->SetBranchAddress(branchName.str().c_str(), &OccupancyPtr_[iHLX], &b_Occupancy[iHLX]);
      
      LHCPtr_[iHLX] = &(lumiSection_->lhc[iHLX]);
      branchName.str(std::string());
      branchName << "LHC" << std::setw(2) << std::setfill('0') << iHLX << ".";
      mChain_->SetBranchAddress(branchName.str().c_str(), &LHCPtr_[iHLX], &b_LHC[iHLX]);
    }

  }

  // OTHER
  if( !bEtSumOnly_ ){
    mChain_->SetBranchAddress("Threshold.",        &Threshold_,       &b_Threshold);
    mChain_->SetBranchAddress("Level1_Trigger.",   &L1Trigger_,       &b_L1Trigger);
    mChain_->SetBranchAddress("HLT.",              &HLT_,             &b_HLT);
    mChain_->SetBranchAddress("Trigger_Deadtime.", &TriggerDeadtime_, &b_TriggerDeadtime);
    mChain_->SetBranchAddress("HF_Ring_Set.",      &RingSet_,         &b_RingSet);
  }
}

unsigned int HCAL_HLX::ROOTFileReader::GetEntries(){

  return mChain_->GetEntries();
}

int HCAL_HLX::ROOTFileReader::GetEntry( int entry ){

  int bytes =  mChain_->GetEntry(entry);
  return bytes;
}

int HCAL_HLX::ROOTFileReader::GetLumiSection( HCAL_HLX::LUMI_SECTION& localSection){

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
