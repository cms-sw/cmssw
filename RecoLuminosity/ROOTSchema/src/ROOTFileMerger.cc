#include "RecoLuminosity/ROOTSchema/interface/ROOTFileMerger.h"
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileWriter.h"
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileReader.h"
 

#include <iostream>
#include <iomanip>
#include <cstdio>

#include <TFile.h>
#include <TChain.h>

HCAL_HLX::ROOTFileMerger::ROOTFileMerger():minSectionNumber_(99999), 
					   RFWriter_(0), 
					   RFReader_(0),
					   lumiSection_(0)
{
    Init();
}

HCAL_HLX::ROOTFileMerger::~ROOTFileMerger(){
  CleanUp();
}

void HCAL_HLX::ROOTFileMerger::Init(){
  RFWriter_ = new ROOTFileWriter;
  if( RFWriter_ == 0 ){
    // Could not allocate memory.
    // Do something.
  }

  RFReader_ = new ROOTFileReader; 
  if( RFReader_ == 0 ){
    // Could not allocate memory.
    // Do something.
  }

  lumiSection_ = new HCAL_HLX::LUMI_SECTION;
  if( lumiSection_ == 0 ){
    // Could not allocate memory.
    // Do something.
  }    
}

void HCAL_HLX::ROOTFileMerger::CleanUp(){

  if( RFWriter_ != 0){
    delete RFWriter_;
  }
  
  if( RFReader_ != 0){
    delete RFReader_;
  }

  if( lumiSection_ !=0 ){
    delete lumiSection_;
  }
}


void HCAL_HLX::ROOTFileMerger::Merge(const unsigned int runNumber, bool bCMSLive){
    /*
       TChain::Merge and TTree::CloneTree leak because we used TTree::Bronch to create the tree.
    */

    RFReader_->SetRunNumber(runNumber);
    unsigned int firstSectionNumber = RFReader_->GetFirstSectionNumber();

    RFWriter_->SetFileName(runNumber, firstSectionNumber);
        
    int nentries = RFReader_->GetNumEntries();
  
    for(int i = 0; i < nentries; i++){
      RFReader_->GetEntry(i, *lumiSection_);

      // Must fill Threshold eventually  right now it contains fake data.
      RFWriter_->FillTree(*lumiSection_);
    }
    
    RFWriter_->CloseTree();
}

void HCAL_HLX::ROOTFileMerger::SetOutputDir(const std::string &dirName){
  RFWriter_->SetDir(dirName);
}

std::string HCAL_HLX::ROOTFileMerger::GetOutputFileName(){
  return RFWriter_->GetFileName();
}

void HCAL_HLX::ROOTFileMerger::SetEtSumOnly(bool bEtSumOnly){
  RFWriter_->SetEtSumOnly(bEtSumOnly);
  RFReader_->SetEtSumOnly(bEtSumOnly);
}
