#include "CondTools/SiPixel/interface/SiPixelGainCalibrationService.h"
#include "FWCore/Utilities/interface/Exception.h"

SiPixelGainCalibrationService::SiPixelGainCalibrationService(const edm::ParameterSet& conf):
  conf_(conf),
  UseCalibDataFromDB_(conf.getParameter<bool>("UseCalibDataFromDB")),
  PedestalValue_(conf.getParameter<double>("PedestalValue")),
  GainValue_(conf.getParameter<double>("GainValue")),
  minGain_(conf.getParameter<double>("MinGain")),
  maxGain_(conf.getParameter<double>("MaxGain")),
  minPed_(conf.getParameter<double>("MinPed")),
  maxPed_(conf.getParameter<double>("MaxPed")),
  ESetupInit_(false)
{

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
    ESetupInit_ = true;
  }
}

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
      return  decodePed(ped->getPed(col, row, old_range, old_cols));
    } else throw cms::Exception("NullPointer")
      << "[SiPixelGainCalibrationService::getPedestal] SiPixelGainCalibrationRcd not initialized ";
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
   if(ESetupInit_) {
     //&&&&&&&&&&&&&&&&&&&&
     //Access from DB
     //&&&&&&&&&&&&&&&&&&&&
     if (detID != old_detID){
       old_detID=detID;
       old_range = ped->getRange(detID);
       old_cols  = ped->getNCols(detID);
     }
     return decodeGain(ped->getGain(col, row, old_range, old_cols));
   } else throw cms::Exception("NullPointer")
     << "[SiPixelGainCalibrationService::getGain] SiPixelGainCalibrationRcd not initialized ";
  }
}

float SiPixelGainCalibrationService::encodeGain( const float& gain ) {
  
  double precision   = (maxGain_-minGain_)/255.;
  float  encodedGain = (float)((gain-minGain_)/precision);
  return encodedGain;

}

float SiPixelGainCalibrationService::encodePed( const float& ped ) {
  
  double precision   = (maxPed_-minPed_)/255.;
  float  encodedPed = (float)((ped-minPed_)/precision);
  return encodedPed;

}

float SiPixelGainCalibrationService::decodePed( const float& ped ) {

  double precision = (maxPed_-minPed_)/255.;
  float decodedPed = (float)(ped*precision + minPed_);
  return decodedPed;

}

float SiPixelGainCalibrationService::decodeGain( const float& gain ) {

  double precision = (maxGain_-minGain_)/255.;
  float decodedGain = (float)(gain*precision + minGain_);
  return decodedGain;

}

