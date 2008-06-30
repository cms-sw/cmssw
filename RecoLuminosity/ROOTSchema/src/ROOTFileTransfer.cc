#include "RecoLuminosity/ROOTSchema/interface/ROOTFileTransfer.h"

HCAL_HLX::ROOTFileTransfer::ROOTFileTransfer(){

  fileName_ = "";
}

HCAL_HLX::ROOTFileTransfer::~ROOTFileTransfer(){
}

void HCAL_HLX::ROOTFileTransfer::SetFileName(std::string fileName ){

  fileName_ = fileName;

}

int HCAL_HLX::ROOTFileTransfer::TransferFile(){

  int errorCode;

  if( fileName_ == "" ){
    // No File set
    errorCode = -1;
  }else{

    // Transfer File
    std::system("/cms/mon/data/dqm/lumi/root/X-Fer/store/lumi/200805/sendNotification-20080508.sh"); 
  }
}
