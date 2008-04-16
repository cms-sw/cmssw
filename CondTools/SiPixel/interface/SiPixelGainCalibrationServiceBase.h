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

// Abstract base class provides common interface to different payload getters 
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

  virtual bool isDead       ( const uint32_t& detID, const int& col, const int& row )=0;
  virtual bool isDeadColumn ( const uint32_t& detID, const int& col, const int& row )=0;

  void    setESObjects(const edm::EventSetup& es );

  std::vector<uint32_t> getDetIds();

 protected:

  float   getPedestalByPixel(const uint32_t& detID,const int& col, const int& row, bool& isDeadPixel) ;
  float   getGainByPixel(const uint32_t& detID,const int& col, const int& row, bool& isDeadPixel) ;

  // the getByColumn functions caches the data to prevent multiple lookups on averaged quanitities
  float   getPedestalByColumn(const uint32_t& detID,const int& col, const int& row, bool& isDeadColumn) ;
  float   getGainByColumn(const uint32_t& detID,const int& col, const int& row, bool& isDeadColumn) ;

  void    throwExepctionForBadRead(std::string payload, const uint32_t& detID, const int& col, const int& row) const;

 private:

  edm::ParameterSet conf_;
  bool ESetupInit_;
  edm::ESHandle<thePayloadObject> ped;
  int numberOfRowsAveragedOver_;

  uint32_t old_detID;
  int      old_cols;
  // Cache data for payloads that average over columns
  
  // these two quantities determine what column averaged block we are in - i.e. ROC 1 or ROC 2
  int      oldAveragedBlockDataGain_;
  int      oldAveragedBlockDataPed_;
  
  bool     oldThisColumnIsDeadGain_;
  bool     oldThisColumnIsDeadPed_;
  int      oldColumnIndexGain_;
  int      oldColumnIndexPed_;
  float    oldColumnValueGain_;
  float    oldColumnValuePed_;

  typename thePayloadObject::Range old_range;
};

template<class thePayloadObject, class theDBRecordType>
SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::SiPixelGainCalibrationServicePayloadGetter(const edm::ParameterSet& conf):
  conf_(conf),
  ESetupInit_(false)
{

  edm::LogInfo("SiPixelGainCalibrationServicePayloadGetter")  << "[SiPixelGainCalibrationServicePayloadGetter::SiPixelGainCalibrationServicePayloadGetter]";
  // Initialize cache variables
  old_detID             = 0;
  oldColumnIndexGain_   = -1;
  oldColumnIndexPed_    = -1;
  oldColumnValueGain_   = 0.;
  oldColumnValuePed_    = 0.; 

  oldAveragedBlockDataGain_ = -1;
  oldAveragedBlockDataPed_  = -1;
  oldThisColumnIsDeadGain_ = false;
  oldThisColumnIsDeadPed_  = false;

}

template<class thePayloadObject, class theDBRecordType>
void SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::setESObjects( const edm::EventSetup& es ) {

    es.get<theDBRecordType>().get(ped);
    numberOfRowsAveragedOver_ = ped->getNumberOfRowsToAverageOver();
    ESetupInit_ = true;

}

template<class thePayloadObject, class theDBRecordType>
std::vector<uint32_t> SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getDetIds() {

  std::vector<uint32_t> vdetId_;  
  ped->getDetIds(vdetId_);
  return vdetId_;

}

template<class thePayloadObject, class theDBRecordType>
float SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getPedestalByPixel(const uint32_t& detID,const int& col, const int& row, bool& isDead) { 
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
      return  ped->getPed(col, row, old_range, old_cols, isDead);
  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationServicePayloadGetter::getPedestalByPixel] SiPixelGainCalibrationRcd not initialized ";
}


template<class thePayloadObject, class theDBRecordType>
float SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getGainByPixel(const uint32_t& detID,const int& col, const int& row, bool& isDead) {
  
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
    return ped->getGain(col, row, old_range, old_cols, isDead);
  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationServicePayloadGetter::getGainByPixel] SiPixelGainCalibrationRcd not initialized ";
}


template<class thePayloadObject, class theDBRecordType>
float SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getPedestalByColumn(const uint32_t& detID,const int& col, const int& row, bool& isDeadColumn) { 
  if(ESetupInit_) {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
      // see if we are in the same averaged data block
      bool inTheSameAveragedDataBlock = false;
      if ( row / numberOfRowsAveragedOver_ == oldAveragedBlockDataPed_ )
         inTheSameAveragedDataBlock = true;

      if (detID != old_detID){
	old_detID=detID;
        std::pair<const typename thePayloadObject::Range, const int> rangeAndNCols = ped->getRangeAndNCols(detID);
	old_range = rangeAndNCols.first;
	old_cols  = rangeAndNCols.second;
      } 
      else if (col == oldColumnIndexPed_ && inTheSameAveragedDataBlock) // same DetID, same column, same data block
      {
         isDeadColumn = oldThisColumnIsDeadPed_;
         return oldColumnValuePed_;
      } 

      oldColumnIndexPed_       = col;
      oldAveragedBlockDataPed_ = row / numberOfRowsAveragedOver_;
      oldColumnValuePed_       = ped->getPed(col, row, old_range, old_cols, isDeadColumn);
      oldThisColumnIsDeadPed_  = isDeadColumn;

      return oldColumnValuePed_;

  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationServicePayloadGetter::getPedestalByColumn] SiPixelGainCalibrationRcd not initialized ";
}


template<class thePayloadObject, class theDBRecordType>
float SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getGainByColumn(const uint32_t& detID,const int& col, const int& row, bool& isDeadColumn) {
  
  if(ESetupInit_) {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
    bool inTheSameAveragedDataBlock = false;
    if ( row / numberOfRowsAveragedOver_ == oldAveragedBlockDataGain_ )
       inTheSameAveragedDataBlock = true;

    if (detID != old_detID){
      old_detID=detID;
      std::pair<const typename thePayloadObject::Range, const int> rangeAndNCols = ped->getRangeAndNCols(detID);
      old_range = rangeAndNCols.first;
      old_cols  = rangeAndNCols.second;
    }
    else if (col == oldColumnIndexGain_ && inTheSameAveragedDataBlock) // same DetID, same column
    {
       isDeadColumn = oldThisColumnIsDeadGain_;
       return oldColumnValueGain_;
    }

    oldColumnIndexGain_       = col;
    oldAveragedBlockDataGain_ = row / numberOfRowsAveragedOver_;
    oldColumnValueGain_       = ped->getGain(col, row, old_range, old_cols, isDeadColumn);
    oldThisColumnIsDeadGain_  = isDeadColumn;

    return oldColumnValueGain_;

  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationServicePayloadGetter::getGainByColumn] SiPixelGainCalibrationRcd not initialized ";
}

template<class thePayloadObject, class theDBRecordType>
void SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::throwExepctionForBadRead(std::string payload, const uint32_t& detID, const int& col, const int& row) const
{
   throw cms::Exception("SiPixelGainCalibration")
      << "[SiPixelGainCalibrationServicePayloadGetter] ERROR - Slow down, speed racer! You have tried to read the ped/gain on a pixel that is flagged as dead. For payload: " << payload << "  DETID: " 
      << detID << " col: " << col << " row: " << row << ". You must check if the pixel is dead before asking for the ped/gain value, otherwise you will get corrupt data!";
}


#endif
