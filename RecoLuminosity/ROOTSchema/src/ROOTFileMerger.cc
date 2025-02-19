#include "RecoLuminosity/ROOTSchema/interface/ROOTFileMerger.h"
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileWriter.h"
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileReader.h"

#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"
 
#include <iomanip>
#include <cstdio>

#include <TFile.h>
#include <TChain.h>

HCAL_HLX::ROOTFileMerger::ROOTFileMerger():minSectionNumber_(99999)
{

  RFWriter_    = NULL;
  RFReader_    = NULL;
  lumiSection_ = NULL;

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

  RFWriter_->SetMerge(true);
}

HCAL_HLX::ROOTFileMerger::~ROOTFileMerger(){

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

bool HCAL_HLX::ROOTFileMerger::Merge(const unsigned int runNumber, const unsigned int minSectionNumber ){

    /*
      TChain::Merge and TTree::CloneTree leak because we used TTree::Bronch to create the tree.
    */
  
  RFReader_->CreateFileNameList();
  
  // RFWriter_->SetFileName(runNumber, firstSectionNumber);
  RFWriter_->OpenFile(runNumber, minSectionNumber);
  
  unsigned int nentries = RFReader_->GetEntries();
  if( nentries == 0 ){
    RFWriter_->CloseFile();
    return false;
  }
  
  for(unsigned int iEntry = 0; iEntry < nentries; ++iEntry ){
    memset( lumiSection_, 0, sizeof(LUMI_SECTION));
    RFReader_->GetEntry(iEntry);
    RFReader_->GetLumiSection(*lumiSection_);
    
    // Must fill Threshold eventually  right now it contains fake data.
    RFWriter_->FillTree(*lumiSection_);
  }
  
  return RFWriter_->CloseFile();
}

void HCAL_HLX::ROOTFileMerger::SetInputDir( const std::string &dirName){

  RFReader_->SetDir(dirName);
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

void HCAL_HLX::ROOTFileMerger::SetFileType( const std::string &fileType ){

  RFWriter_->SetFileType( fileType );
  RFReader_->SetFileType( fileType );
}

void HCAL_HLX::ROOTFileMerger::SetDate( const std::string &date ){
  
  RFWriter_->SetDate( date );
  RFReader_->SetDate( date );
}  
