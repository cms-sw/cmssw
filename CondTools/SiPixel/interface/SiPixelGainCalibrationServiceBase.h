#ifndef CondTools_SiPixel_SiPixelGainCalibrationServiceBase_H
#define CondTools_SiPixel_SiPixelGainCalibrationServiceBase_H

// ************************************************************************
// ************************************************************************
// *******     SiPixelOfflineCalibrationServiceBase                 *******
// *******     Author: Vincenzo Chiochia (chiochia@cern.ch)         *******
// *******     Modified: Evan Friis (evan.friis@cern.ch)            *******
// *******                                                          *******
// *******     Provides common interface to SiPixel gain calib      *******
// *******     payloads in offline database                         *******
// *******                                                          *******
// ************************************************************************
// ************************************************************************



// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <utility>

// Abstract base class provides common to different payload getters 
class SiPixelGainCalibrationServiceBase {
   public:
      SiPixelGainCalibrationServiceBase(){};
      virtual ~SiPixelGainCalibrationServiceBase(){};
      virtual float getGain(const uint32_t& detID, const int& col, const int& row)=0;
      virtual float getPedestal(const uint32_t& detID, const int& col, const int& row)=0;
      virtual void  setESObjects(const edm::EventSetup& es )=0;
      virtual std::vector<uint32_t> getDetIds()=0;
};


// Abstract template class that defines DB access types and payload specific getters
template<class thePayloadObject, class theDBRecordType>
class SiPixelGainCalibrationServicePayloadGetter : public SiPixelGainCalibrationServiceBase {

 public:
  explicit SiPixelGainCalibrationServicePayloadGetter(const edm::ParameterSet& conf);
  virtual ~SiPixelGainCalibrationServicePayloadGetter(){};

  //Abstract methods
  virtual float getGain(const uint32_t& detID, const int& col, const int& row)=0;
  virtual float getPedestal(const uint32_t& detID, const int& col, const int& row)=0;

  void    setESObjects(const edm::EventSetup& es );

  std::vector<uint32_t> getDetIds();

 protected:

  float   getPedestalByPixel(const uint32_t& detID,const int& col, const int& row) ;
  float   getGainByPixel(const uint32_t& detID,const int& col, const int& row) ;
  float   getPedestalByColumn(const uint32_t& detID,const int& col) ;
  float   getGainByColumn(const uint32_t& detID,const int& col) ;

 private:

  edm::ParameterSet conf_;
  bool ESetupInit_;
  edm::ESHandle<thePayloadObject> ped;

  uint32_t old_detID;
  int      old_cols;
  typename thePayloadObject::Range old_range;
};

template<class thePayloadObject, class theDBRecordType>
SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::SiPixelGainCalibrationServicePayloadGetter(const edm::ParameterSet& conf):
  conf_(conf),
  ESetupInit_(false)
{

  edm::LogInfo("SiPixelGainCalibrationServicePayloadGetter")  << "[SiPixelGainCalibrationServicePayloadGetter::SiPixelGainCalibrationServicePayloadGetter]";
  old_detID = 0;

}

template<class thePayloadObject, class theDBRecordType>
void SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::setESObjects( const edm::EventSetup& es ) {

    es.get<theDBRecordType>().get(ped);
    ESetupInit_ = true;

}

template<class thePayloadObject, class theDBRecordType>
std::vector<uint32_t> SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getDetIds() {

  std::vector<uint32_t> vdetId_;  
  ped->getDetIds(vdetId_);
  return vdetId_;

}

template<class thePayloadObject, class theDBRecordType>
float SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getPedestalByPixel(const uint32_t& detID,const int& col, const int& row) { 
  if(ESetupInit_) {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
      if (detID != old_detID){
	old_detID=detID;
        std::pair<const typename thePayloadObject::Range, const int> rangeAndNCols = ped->getRangeAndNCols(detID);
	old_range = rangeAndNCols.first;
	old_cols  = rangeAndNCols.second;
      }
      //std::cout<<" Pedestal "<<ped->getPed(col, row, old_range, old_cols)<<std::endl;
      return  ped->getPed(col, row, old_range, old_cols);
  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationServicePayloadGetter::getPedestalByPixel] SiPixelGainCalibrationRcd not initialized ";
}


template<class thePayloadObject, class theDBRecordType>
float SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getGainByPixel(const uint32_t& detID,const int& col, const int& row) {
  
  if(ESetupInit_) {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
    if (detID != old_detID){
      old_detID=detID;
      std::pair<const typename thePayloadObject::Range, const int> rangeAndNCols = ped->getRangeAndNCols(detID);
      old_range = rangeAndNCols.first;
      old_cols  = rangeAndNCols.second;
    }
    return ped->getGain(col, row, old_range, old_cols);
  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationServicePayloadGetter::getGainByPixel] SiPixelGainCalibrationRcd not initialized ";
}


template<class thePayloadObject, class theDBRecordType>
float SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getPedestalByColumn(const uint32_t& detID,const int& col) { 
  if(ESetupInit_) {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
      if (detID != old_detID){
	old_detID=detID;
        std::pair<const typename thePayloadObject::Range, const int> rangeAndNCols = ped->getRangeAndNCols(detID);
	old_range = rangeAndNCols.first;
	old_cols  = rangeAndNCols.second;
      }
      //std::cout<<" Pedestal "<<ped->getPed(col, row, old_range, old_cols)<<std::endl;
      return  ped->getPed(col, old_range, old_cols);
  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationServicePayloadGetter::getPedestalByColumn] SiPixelGainCalibrationRcd not initialized ";
}


template<class thePayloadObject, class theDBRecordType>
float SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getGainByColumn(const uint32_t& detID,const int& col) {
  
  if(ESetupInit_) {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
    if (detID != old_detID){
      old_detID=detID;
      std::pair<const typename thePayloadObject::Range, const int> rangeAndNCols = ped->getRangeAndNCols(detID);
      old_range = rangeAndNCols.first;
      old_cols  = rangeAndNCols.second;
    }
    return ped->getGain(col, old_range, old_cols);
  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationServicePayloadGetter::getGainByColumn] SiPixelGainCalibrationRcd not initialized ";
}



#endif
