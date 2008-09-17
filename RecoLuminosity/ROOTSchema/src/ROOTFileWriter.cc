#include "RecoLuminosity/ROOTSchema/interface/ROOTFileWriter.h"

#include <sstream>
#include <iostream>
#include <typeinfo>
#include <iomanip>
#include <vector>
#include <ctime>

#include <stddef.h>

// mkdir
#include <sys/types.h>
#include <sys/stat.h>

#include <TROOT.h>
#include <TChain.h>
#include <TTree.h>
#include <TFile.h>

HCAL_HLX::ROOTFileWriter::ROOTFileWriter(){}

HCAL_HLX::ROOTFileWriter::~ROOTFileWriter(){}

void HCAL_HLX::ROOTFileWriter::CreateTree(){

  if(fileName_ == ""){
    std::cout << "*** File Name was not set ***" << std::endl;
    CleanUp();
    exit(1);
  }

  std::string FullPath;

  FullPath = dirName_;
//   if( !endOfRun_ ){ 
//     FullPath += runDir_;
//     endOfRun_ = false;
//   }
  FullPath +=  fileName_;
  
  m_file = new TFile(FullPath.c_str(), "RECREATE");

  if(!m_file){
    std::cout << " *** Couldn't make or open file: " << fileName_ << " *** " << std::endl;
    CleanUp();
    exit(1);
  }

  m_file->cd();
  
  m_tree   = new TTree("LumiTree","");

  Header_  = &(lumiSection_->hdr);
  Summary_ = &(lumiSection_->lumiSummary);
  Detail_  = &(lumiSection_->lumiDetail);
  
  m_tree->Bronch("Header.",  "HCAL_HLX::LUMI_SECTION_HEADER", &Header_,  1);

  if( !bEtSumOnly_ ){
    m_tree->Bronch("Summary.", "HCAL_HLX::LUMI_SUMMARY",        &Summary_, 1);
    m_tree->Bronch("Detail.",  "HCAL_HLX::LUMI_DETAIL",         &Detail_,  1);
    
    m_tree->Bronch("Threshold.",        "HCAL_HLX::LUMI_THRESHOLD",   &Threshold_, 1);
    m_tree->Bronch("Level1_Trigger.",   "HCAL_HLX::LEVEL1_TRIGGER",   &L1Trigger_, 1);
    m_tree->Bronch("HLT.",              "HCAL_HLX::HLT",              &HLT_,       1);
    m_tree->Bronch("Trigger_Deadtime.", "HCAL_HLX::TRIGGER_DEADTIME", &TriggerDeadtime_, 1);
    m_tree->Bronch("HF_Ring_Set.",      "HCAL_HLX::LUMI_HF_RING_SET", &RingSet_,1);
  }

  for( unsigned int iHLX = 0; iHLX < 36; ++iHLX ){
    EtSumPtr_[iHLX] = &( lumiSection_->etSum[iHLX] );
    MakeBranch(lumiSection_->etSum[iHLX], &EtSumPtr_[iHLX], iHLX);
    
    if( !bEtSumOnly_ ){
      OccupancyPtr_[iHLX] = &(lumiSection_->occupancy[iHLX]);
      MakeBranch(lumiSection_->occupancy[iHLX], &OccupancyPtr_[iHLX], iHLX);
      
      LHCPtr_[iHLX] = &(lumiSection_->lhc[iHLX]);
      MakeBranch(lumiSection_->lhc[iHLX], &LHCPtr_[iHLX], iHLX);
    }
  }

}

void HCAL_HLX::ROOTFileWriter::FillTree(const HCAL_HLX::LUMI_SECTION& localSection){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  if(fileName_ == ""){
    std::cout << "*** File Name was not set ***" << std::endl;
    return;
  }
  
  memcpy( lumiSection_, &localSection, sizeof(HCAL_HLX::LUMI_SECTION));

  InsertInformation(); // To be modified later.

  m_tree->Fill();

#ifdef DEBUG
  //  m_tree->Print();
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

template< class T >
void HCAL_HLX::ROOTFileWriter::MakeBranch(const T &in, T **out, int HLXNum){
  
  const std::string typeName = typeid(T).name();
  std::string className;
  std::string branchName;
  std::ostringstream numString;

  if(typeName == "N8HCAL_HLX11LHC_SECTIONE"){
    className = "HCAL_HLX::LHC_SECTION";
    branchName = "LHC";
  }else if(typeName == "N8HCAL_HLX17OCCUPANCY_SECTIONE"){
    className = "HCAL_HLX::OCCUPANCY_SECTION";
    branchName = "Occupancy";
  }else if(typeName == "N8HCAL_HLX14ET_SUM_SECTIONE"){
    className = "HCAL_HLX::ET_SUM_SECTION";
    branchName = "ETSum";
  }
  
  numString << std::setfill('0') << std::setw(2) << HLXNum;
  branchName = branchName + numString.str() + ".";
  m_tree->Bronch(branchName.c_str(), className.c_str(), out, 1);

}

void HCAL_HLX::ROOTFileWriter::InsertInformation(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  // This information will eventually come from the lms cell
  Threshold_->OccThreshold1Set1 = 51;
  Threshold_->OccThreshold2Set1 = 52;
  Threshold_->OccThreshold1Set2 = 53;
  Threshold_->OccThreshold2Set2 = 54;
  Threshold_->ETSum             = 55;
  
  L1Trigger_->L1lineNumber  = 71;
  L1Trigger_->L1Scaler      = 72;
  L1Trigger_->L1RateCounter = 73;
  
  HLT_->TriggerPath    = 81;
  HLT_->InputCount     = 82;
  HLT_->AcceptCount    = 83;
  HLT_->PrescaleFactor = 84;

  TriggerDeadtime_->TriggerDeadtime = 91;

  RingSet_->Set1Rings = "33L,34L";
  RingSet_->Set2Rings = "35S,36S";
  RingSet_->EtSumRings = "33L,34L,35S,36S";
}

void HCAL_HLX::ROOTFileWriter::CloseTree(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  if(fileName_ == ""){
    std::cout << "*** File Name was not set ***" << std::endl;
    return;
  }

  m_file->Write();
  m_file->Close();

  if(m_file != NULL){
    delete m_file;
  //delete m_tree; // NO!!! root does this when you delete m_file
    m_file = NULL;
    m_tree = NULL;
  }

}
