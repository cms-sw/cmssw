#include "CondTools/SiPixel/interface/SiPixelGainCalibrationService.h"
#include "FWCore/Utilities/interface/Exception.h"

SiPixelGainCalibrationService::SiPixelGainCalibrationService(const edm::ParameterSet& conf):
  conf_(conf),
  ESetupInit_(false)
{

  edm::LogInfo("SiPixelGainCalibrationService")  << "[SiPixelGainCalibrationService::SiPixelGainCalibrationService]";
  old_detID = 0;

}

void SiPixelGainCalibrationService::setESObjects( const edm::EventSetup& es ) {

    es.get<SiPixelGainCalibrationRcd>().get(ped);
    ESetupInit_ = true;

}

std::vector<uint32_t> SiPixelGainCalibrationService::getDetIds() {

  std::vector<uint32_t> vdetId_;  
  ped->getDetIds(vdetId_);
  return vdetId_;

}

float SiPixelGainCalibrationService::getPedestal (const uint32_t& detID,const int& col, const int& row) {
  
  if(ESetupInit_) {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
      if (detID != old_detID){
	old_detID=detID;
	old_range = ped->getRange(detID);
	old_cols  = ped->getNCols(detID);
      }
      //std::cout<<" Pedestal "<<ped->getPed(col, row, old_range, old_cols)<<std::endl;
      return  ped->getPed(col, row, old_range, old_cols);
  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationService::getPedestal] SiPixelGainCalibrationRcd not initialized ";
}


float SiPixelGainCalibrationService::getGain (const uint32_t& detID,const int& col, const int& row) {
  
  if(ESetupInit_) {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
    if (detID != old_detID){
      old_detID=detID;
      old_range = ped->getRange(detID);
       old_cols  = ped->getNCols(detID);
    }
    return ped->getGain(col, row, old_range, old_cols);
  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationService::getGain] SiPixelGainCalibrationRcd not initialized ";
}

