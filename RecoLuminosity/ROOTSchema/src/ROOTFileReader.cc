#include "RecoLuminosity/ROOTSchema/interface/ROOTFileReader.h"

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
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
  
  mChain_ = new TChain("LumiTree");
}

HCAL_HLX::ROOTFileReader::~ROOTFileReader(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  delete mChain_;  
}

bool HCAL_HLX::ROOTFileReader::SetRunNumber( const unsigned int runNumber, const std::string &month){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  std::stringstream runDir;
  runDir.str(std::string());
  
  std::string month_;

  if( month == "" ){
    month_ = TimeStampYYYYMM();
  }else{
    month_ = month;
  }

  runDir << dirName_ << "/" << month << "/" << std::setw(9) << std::setfill('0') << runNumber << "/";

  if( opendir( runDir.str().c_str()) == NULL ){
    return false;
  }

  return CreateFileNameList( runDir.str() );
}

int HCAL_HLX::ROOTFileReader::CreateFileNameList( const std::string &runDir ){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  DIR *dp;
  struct dirent *dirp;
  std::string tempFileName;

  std::vector< std::string > fileNames;

  fileNames.clear();

  if( runDir == ""){
    return false;
  }

  if( ( dp = opendir( runDir.c_str() )  ) == NULL ){
    closedir(dp);
    return false;
  }

  while( (dirp = readdir(dp)) != NULL ){
    tempFileName = dirp->d_name;
    if(tempFileName.substr(0,8) == "CMS_LUMI" ){
      fileNames.push_back(runDir + "/" + tempFileName);
    }
  }
  closedir(dp);

  if( fileNames.size() == 0 ){
    return 0;
  }

  sort(fileNames.begin(), fileNames.end());
  return ReplaceFile( fileNames );
}

int HCAL_HLX::ROOTFileReader::ReplaceFile(const std::string &fileName){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
  
  std::vector< std::string > tempVecOfStrings;

  tempVecOfStrings.clear();
  tempVecOfStrings.push_back(fileName);
  return ReplaceFile( tempVecOfStrings );

}

int HCAL_HLX::ROOTFileReader::ReplaceFile(const std::vector< std::string> &fileNames){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  // replacing a file changes the run number and section number automatically.
  
  mChain_->Reset();

  for( std::vector< std::string >::const_iterator VoS = fileNames.begin(); VoS != fileNames.end(); ++VoS){
    mChain_->Add((*VoS).c_str());
  }

  numEntries_ = mChain_->GetEntries();

  return 0;
}

void HCAL_HLX::ROOTFileReader::CreateTree(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  Header_  = &(lumiSection_->hdr);
  Summary_ = &(lumiSection_->lumiSummary);
  Detail_  = &(lumiSection_->lumiDetail);

  mChain_->SetBranchAddress("Header.",  &Header_,  &b_Header);

  mChain_->SetBranchAddress("Summary.", &Summary_, &b_Summary);
  mChain_->SetBranchAddress("Detail.",  &Detail_,  &b_Detail);


  for(unsigned int iHLX = 0; iHLX < 36; ++iHLX){
    std::stringstream branchName;

    EtSumPtr_[iHLX] = &(lumiSection_->etSum[iHLX]);
    branchName.str(std::string());
    branchName << "ETSum" << std::setw(2) << std::setfill('0') << iHLX << ".";
    mChain_->SetBranchAddress(branchName.str().c_str(), &EtSumPtr_[iHLX], &b_ETSum[iHLX]);

    OccupancyPtr_[iHLX] = &(lumiSection_->occupancy[iHLX]);
    branchName.str(std::string());
    branchName << "Occupancy" << std::setw(2) << std::setfill('0') << iHLX << ".";
    mChain_->SetBranchAddress(branchName.str().c_str(), &OccupancyPtr_[iHLX], &b_Occupancy[iHLX]);
    
    LHCPtr_[iHLX] = &(lumiSection_->lhc[iHLX]);
    branchName.str(std::string());
    branchName << "LHC" << std::setw(2) << std::setfill('0') << iHLX << ".";
    mChain_->SetBranchAddress(branchName.str().c_str(), &LHCPtr_[iHLX], &b_LHC[iHLX]);
    
  }

  // OTHER
  mChain_->SetBranchAddress("Threshold.",        &Threshold_,       &b_Threshold);
  mChain_->SetBranchAddress("Level1_Trigger.",   &L1Trigger_,       &b_L1Trigger);
  mChain_->SetBranchAddress("HLT.",              &HLT_,             &b_HLT);
  mChain_->SetBranchAddress("Trigger_Deadtime.", &TriggerDeadtime_, &b_TriggerDeadtime);
  mChain_->SetBranchAddress("HF_Ring_Set.",      &RingSet_,         &b_RingSet);


}

int HCAL_HLX::ROOTFileReader::GetNumEntries(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  return numEntries_;
}

int HCAL_HLX::ROOTFileReader::GetEntry(int entry, HCAL_HLX::LUMI_SECTION& localSection){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  int bytes =  mChain_->GetEntry(entry);
  memcpy(&localSection, lumiSection_, sizeof(HCAL_HLX::LUMI_SECTION));
  return bytes;
}

int HCAL_HLX::ROOTFileReader::GetThreshold(HCAL_HLX::LUMI_THRESHOLD&  localThreshold){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  memcpy(&localThreshold, Threshold_, sizeof(localThreshold));
  return 0;
}

int HCAL_HLX::ROOTFileReader::GetHFRingSet(HCAL_HLX::LUMI_HF_RING_SET& localRingSet){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  memcpy(&localRingSet, RingSet_, sizeof(localRingSet));
  return 0;
}

int HCAL_HLX::ROOTFileReader::GetL1Trigger(HCAL_HLX::LEVEL1_TRIGGER& localL1Trigger){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  memcpy(&localL1Trigger, L1Trigger_, sizeof(localL1Trigger));
  return 0;
}

int HCAL_HLX::ROOTFileReader::GetHLT(HCAL_HLX::HLT& localHLT){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  memcpy(&localHLT, HLT_, sizeof(localHLT));
  return 0;
}

int HCAL_HLX::ROOTFileReader::GetTriggerDeadtime(HCAL_HLX::TRIGGER_DEADTIME& localTD){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  memcpy(&localTD, TriggerDeadtime_, sizeof(localTD));
  return 0;
}
