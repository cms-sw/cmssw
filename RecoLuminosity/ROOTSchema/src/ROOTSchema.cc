/*

Author: Adam Hunt - Princeton University
Date: 2007-10-05

This is the control layer of the root file writer.  This is where all the error checking takes place. 
The worker layer does not do any error checking.  This is called by the user layer.

*/

//
// constructor and destructor
//

#include "RecoLuminosity/ROOTSchema/interface/ROOTSchema.h"
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileWriter.h"
#include "RecoLuminosity/ROOTSchema/interface/HTMLGenerator.hh"
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileMerger.h"
#include "RecoLuminosity/ROOTSchema/interface/ROOTFileTransfer.h"

#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

#include <iomanip>

HCAL_HLX::ROOTSchema::ROOTSchema():RFMerger_(NULL), RFTransfer_(NULL),
				   LumiHTML_(NULL), RFWriter_(NULL),
				   previousRun_(0), firstSectionNumber_(0),
				   startTime_(0),
				   bMerge_(0), bWBM_(0), bTransfer_(0),
				   bEtSumOnly_(false),
				   fileType_("RAW"),
				   lsDir_(""),
				   mergeDir_(""),
				   dateDir_(""),
				   runDir_("")
{

  // Allocate memory for private members.
  RFWriter_   = new ROOTFileWriter;
  if( !RFWriter_ ){
    //Could not allocate memory.
  }
  RFMerger_   = new ROOTFileMerger;
  if( !RFMerger_ ){
    //Could not allocate memory.
  }
  RFTransfer_ = new ROOTFileTransfer;
  if( !RFTransfer_ ){
    // Could not allocate memory.
  }
  LumiHTML_ = new HTMLGenerator;
  if( !LumiHTML_ ){
    // Could not allocate memory.
  }
}

HCAL_HLX::ROOTSchema::~ROOTSchema(){
  // Deallocate memory for private members.

  EndRun();

  if( RFWriter_ ){
    delete RFWriter_;
    RFWriter_ = 0;
  }

  if( RFMerger_ ){
    delete RFMerger_;
    RFMerger_ = 0;
  }

  if( RFTransfer_ ){
    delete RFTransfer_;
    RFTransfer_ = 0;
  }

  if( LumiHTML_ ){
    delete LumiHTML_;
    LumiHTML_ = 0;
  }
}

// *************************************** Configuration ************************************

// ******** General *********** 

void HCAL_HLX::ROOTSchema::SetEtSumOnly(const bool bEtSumOnly){
  
  bEtSumOnly_ = bEtSumOnly;
}

void HCAL_HLX::ROOTSchema::SetFileType( const std::string &fileType ){

  if( fileType_ != fileType ){
    fileType_ = fileType;
    if(  (fileType_ != "RAW") && (fileType_ != "VDM") && (fileType_ != "ET") ){
      fileType_ = "RAW";  // Default to RAW.
    }
    
    if( fileType_ != "ET" && bEtSumOnly_ ){
      bEtSumOnly_ = false;
    }
    if( fileType_ == "ET" && !bEtSumOnly_ ){
      bEtSumOnly_ = true;
    }

    RFWriter_->SetEtSumOnly( bEtSumOnly_ );
    LumiHTML_->SetEtSumOnly( bEtSumOnly_ );
    RFMerger_->SetEtSumOnly( bEtSumOnly_ );
    RFTransfer_->SetEtSumOnly( bEtSumOnly_ );

    RFWriter_->SetFileType(fileType_);
    LumiHTML_->SetFileType(fileType_);
    RFMerger_->SetFileType(fileType_);
    RFTransfer_->SetFileType(fileType_);

  }

}

// ******** LS Writer *********

void HCAL_HLX::ROOTSchema::SetLSDir(const std::string &lsDir ){
  
  lsDir_ = lsDir;
  if( lsDir_.substr( lsDir_.size() - 1) != "/" ){
    lsDir_ += "/";
  }
}

// ******** Merger ************

void HCAL_HLX::ROOTSchema::SetMergeFiles(bool bMerge){
  
  bMerge_ = bMerge;
  if( !bMerge_ ){
    bTransfer_ = false;
  } 
}

void HCAL_HLX::ROOTSchema::SetMergeDir(const std::string &mergeDir ){

  mergeDir_ = mergeDir;
  if( mergeDir_.substr( mergeDir_.size() - 1) != "/" ){
    mergeDir_ += "/";
  }
}

// ******** Transfer **********

void HCAL_HLX::ROOTSchema::SetTransferFiles(const bool bTransfer){

  bTransfer_ = bTransfer;
  if( bTransfer_ ){
    bMerge_ = true;
  }
}

// ******** HTML **************

void HCAL_HLX::ROOTSchema::SetCreateWebPage(const bool bWBM){

  bWBM_ = bWBM;
}

void HCAL_HLX::ROOTSchema::SetWebDir(const std::string &webDir){

  LumiHTML_->SetOutputDir( webDir );
}

void HCAL_HLX::ROOTSchema::SetHistoBins(const int NBins, const double XMin, const double XMax){

  LumiHTML_->SetHistoBins( NBins, XMin, XMax );
}

// *********************************** Handle Lumi section and end of run ************************

bool HCAL_HLX::ROOTSchema::ProcessSection(const HCAL_HLX::LUMI_SECTION &lumiSection){

  if( (previousRun_) != (lumiSection.hdr.runNumber) ){
    EndRun();
    
    // Keep track of run information.
    previousRun_        = lumiSection.hdr.runNumber;
    firstSectionNumber_ = lumiSection.hdr.sectionNumber;
    startTime_          = lumiSection.hdr.timestamp;

    RFWriter_->SetDate( TimeStampYYYYMMDD( startTime_ ));
    RFMerger_->SetDate( TimeStampYYYYMMDD( startTime_ ));
     
    // Create directory structure for the new run.
    std::stringstream runDirss;
    runDirss.str(std::string(""));
    dateDir_ = TimeStampYYYYMM( startTime_ ) + "/";
    runDirss << std::setw(9) << std::setfill('0') << previousRun_ << "/";
    runDir_ = runDirss.str();

    MakeDir( lsDir_ + dateDir_ + runDir_ , 0775);
    RFWriter_->SetDir( lsDir_ + dateDir_ + runDir_ );

    RFMerger_->SetInputDir(  lsDir_ + dateDir_ + runDir_ );
    MakeDir( mergeDir_ + dateDir_ , 0775);
    RFMerger_->SetOutputDir( mergeDir_ + dateDir_ );

    LumiHTML_->SetInputDir(  lsDir_ + dateDir_ + runDir_ );
    RFTransfer_->SetInputDir( mergeDir_ + dateDir_ );
  }

  // Write individual lumi section files.
  RFWriter_->OpenFile( lumiSection );
  RFWriter_->FillTree( lumiSection );
  RFWriter_->CloseFile();

  // Create a web page.
  if( bWBM_ ){
    LumiHTML_->CreateWebPage( RFWriter_->GetFileName(), 0 );
  }
  return true;
}

void HCAL_HLX::ROOTSchema::EndRun(){

  if( previousRun_ != 0 ){
    if( bMerge_ ){
      RFMerger_->Merge( previousRun_, firstSectionNumber_ );
      
      if( bTransfer_ ){
	RFTransfer_->SetFileName( RFMerger_->GetOutputFileName());
	RFTransfer_->TransferFile();
      }
    }
    previousRun_ = 0;
  }
}
