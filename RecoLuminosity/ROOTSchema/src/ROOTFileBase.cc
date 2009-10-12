#include "RecoLuminosity/ROOTSchema/interface/ROOTFileBase.h"

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

HCAL_HLX::ROOTFileBase::ROOTFileBase():filePrefix_("CMS_LUMI_RAW"),
				       dirName_("./"),
				       bEtSumOnly_(false),
				       date_(""),
				       fileType_("RAW")
{}

HCAL_HLX::ROOTFileBase::~ROOTFileBase(){}

void HCAL_HLX::ROOTFileBase::Init(){

  lumiSection_     = new HCAL_HLX::LUMI_SECTION;

  Threshold_       = new HCAL_HLX::LUMI_THRESHOLD;
  L1Trigger_       = new HCAL_HLX::LEVEL1_TRIGGER;
  HLT_             = new HCAL_HLX::HLT;
  TriggerDeadtime_ = new HCAL_HLX::TRIGGER_DEADTIME;
  RingSet_         = new HCAL_HLX::LUMI_HF_RING_SET;
}

void HCAL_HLX::ROOTFileBase::CleanUp(){

  delete lumiSection_;
  
  delete Threshold_;
  delete L1Trigger_;
  delete HLT_;
  delete TriggerDeadtime_; 
  delete RingSet_;
}

void HCAL_HLX::ROOTFileBase::SetDir(const std::string& dirName){

  dirName_ = dirName;
}

void HCAL_HLX::ROOTFileBase::SetFileType(const std::string &fileType){
  
  fileType_ = fileType;
}

void HCAL_HLX::ROOTFileBase::SetDate(const std::string &date){

  date_ = date;
}

void HCAL_HLX::ROOTFileBase::SetFileName(const unsigned int runNumber, 
					 const unsigned int sectionNumber ){
  
  std::stringstream fileName;
  fileName.str(std::string());
  
  fileName << "CMS_LUMI_" << fileType_ << "_" << date_ << "_"
	   << std::setfill('0') << std::setw(9) << runNumber << "_"
	   << std::setfill('0') << std::setw(4) << sectionNumber << ".root";
  
  fileName_ = fileName.str();
  
  CreateTree();
}

void HCAL_HLX::ROOTFileBase::SetFileName(const HCAL_HLX::LUMI_SECTION &lumiSection){

  SetFileName( lumiSection.hdr.runNumber,
	       lumiSection.hdr.sectionNumber);
}

void HCAL_HLX::ROOTFileBase::SetEtSumOnly( bool bEtSumOnly ){

  bEtSumOnly_ = bEtSumOnly;
} 
