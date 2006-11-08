#include "CondTools/SiPixel/interface/SiPixelGainCalibrationService.h"

SiPixelGainCalibrationService::SiPixelGainCalibrationService(const edm::ParameterSet& conf):
  conf_(conf),
  UseCalibDataFromDB_(conf.getParameter<bool>("UseCalibDataFromDB")) {

  if (UseCalibDataFromDB_==false){  
    edm::LogInfo("SiPixelGainCalibrationService")  << "[SiPixelGainCalibrationService::SiPixelGainCalibrationService] Using a Single Value for Pedestal and Gain";
  } 
  else {
    edm::LogInfo("SiPixelGainCalibrationService")  << "[SiPixelGainCalibrationService::SiPixelGainCalibrationService] Using Calibration Data from DB";
  }

  old_detID = 0;

}

void SiPixelGainCalibrationService::setESObjects( const edm::EventSetup& es ) {
  if ( UseCalibDataFromDB_ == true ) {
    es.get<SiPixelGainCalibrationRcd>().get(ped);
  }
};

std::vector<uint32_t> SiPixelGainCalibrationService::getDetIds() {

  std::vector<uint32_t> vdetId_;

  if ( UseCalibDataFromDB_ == true ) {
    ped->getDetIds(vdetId_);
  }

  return vdetId_;

}

float SiPixelGainCalibrationService::getPedestal (const uint32_t& detID,const int& col, const int& row) {

  if (UseCalibDataFromDB_==false){  
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //Case of SingleValue of Pedestals for all pixels
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    return (float) PedestalValue_;
  } 
  else {
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

  }
}

float SiPixelGainCalibrationService::getGain (const uint32_t& detID,const int& col, const int& row) {

  if (UseCalibDataFromDB_==false){  
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    //Case of SingleValue of Gain for all pixels
    //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    return (float) GainValue_;
  } 
  else {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
    if (detID != old_detID){
      old_detID=detID;
      old_range = ped->getRange(detID);
      old_cols  = ped->getNCols(detID);
    }
    return ped->getGain(col, row, old_range, old_cols);
  }
}
