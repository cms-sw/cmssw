#include "RecoLuminosity/ROOTSchema/interface/ROOTFileTransfer.h"
#include <sstream>
#include <iostream>

HCAL_HLX::ROOTFileTransfer::ROOTFileTransfer(){

  fileName_ = "";

  dirName_ = "";

}

HCAL_HLX::ROOTFileTransfer::~ROOTFileTransfer(){
}

void HCAL_HLX::ROOTFileTransfer::SetFileName(std::string fileName ){

   fileName_ = fileName;

   dirName_ = "/cms/mon/data/dqm/lumi/root/store/lumi/200807";  // FIX

}

int HCAL_HLX::ROOTFileTransfer::TransferFile(){

  int errorCode;
  std::stringstream commandLine;

  std::cout << "fileName: " << fileName_ << std::endl;
  std::cout << "dirName: " << dirName_ << std::endl;

  if( fileName_ == "" ){
    // No File set
    errorCode = -1;
  }else{

    //Transfer File to Offline DB
    commandLine.str(std::string());
    commandLine << "xferWrapper.sh " << dirName_ << " " << fileName_;
    std::system(commandLine.str().c_str()); 
    
  }
  return 0;

}
