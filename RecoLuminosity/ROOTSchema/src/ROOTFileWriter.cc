#include "RecoLuminosity/ROOTSchema/interface/ROOTFileWriter.h"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

#include <sstream>
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

HCAL_HLX::ROOTFileWriter::ROOTFileWriter():bMerge_(false){

  Init(); // ROOTFileBase
}

HCAL_HLX::ROOTFileWriter::~ROOTFileWriter(){

  CleanUp(); // ROOTFileBase
}

bool HCAL_HLX::ROOTFileWriter::OpenFile(const HCAL_HLX::LUMI_SECTION &localSection){

  return OpenFile( localSection.hdr.runNumber,
		   localSection.hdr.sectionNumber);
}

bool HCAL_HLX::ROOTFileWriter::OpenFile( const unsigned int runNumber, 
					 const unsigned int sectionNumber){

  SetFileName( runNumber, sectionNumber);

  m_file = new TFile( (dirName_ + fileName_).c_str(), "RECREATE");
  if( !m_file ){
    return false;
  }

  m_file->cd();
  CreateTree();

  return true;
}

void HCAL_HLX::ROOTFileWriter::CreateTree(){
  
  m_tree   = new TTree("LumiTree","");

  Header_  = &(lumiSection_->hdr);
  Summary_ = &(lumiSection_->lumiSummary);
  Detail_  = &(lumiSection_->lumiDetail);
  
  m_tree->Bronch("Header.",  "HCAL_HLX::LUMI_SECTION_HEADER", &Header_,  1);

  if( !bEtSumOnly_ ){
    m_tree->Bronch("Summary.", "HCAL_HLX::LUMI_SUMMARY",        &Summary_, 1);
    m_tree->Bronch("Detail.",  "HCAL_HLX::LUMI_DETAIL",         &Detail_,  1);
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

template< class T >
void HCAL_HLX::ROOTFileWriter::MakeBranch(const T &in, T **out, const int HLXNum){
  
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

void HCAL_HLX::ROOTFileWriter::FillTree(const HCAL_HLX::LUMI_SECTION& localSection){

  memcpy( lumiSection_, &localSection, sizeof(HCAL_HLX::LUMI_SECTION));
  m_tree->Fill();
}

bool HCAL_HLX::ROOTFileWriter::CloseFile(){
  
  m_tree->Write();
  m_file->Close();
  
  if(m_file != NULL){
    delete m_file;
    //delete m_tree; // NO!!! root does this when you delete m_file
    m_file = NULL;
    m_tree = NULL;
  }
  
  return true;
}
