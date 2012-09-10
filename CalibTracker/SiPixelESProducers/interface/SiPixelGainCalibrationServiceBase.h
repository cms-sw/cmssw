#ifndef CalibTracker_SiPixelESProducers_SiPixelGainCalibrationServiceBase_H
#define CalibTracker_SiPixelESProducers_SiPixelGainCalibrationServiceBase_H

// ************************************************************************
// ************************************************************************
// *******     SiPixelOfflineCalibrationServiceBase                 *******
// *******     Author: Vincenzo Chiochia (chiochia@cern.ch)         *******
// *******     Modified: Evan Friis (evan.friis@cern.ch)            *******
// *******     Additions: Freya Blekman (freya.blekman@cern.ch)     *******
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
#include <iostream>
#include <utility>

// Abstract base class provides common interface to different payload getters 
class SiPixelGainCalibrationServiceBase {
   public:
      SiPixelGainCalibrationServiceBase(){};
      virtual ~SiPixelGainCalibrationServiceBase(){};
      virtual float getGain      ( const uint32_t& detID , const int& col , const int& row)=0;
      virtual float getPedestal  ( const uint32_t& detID , const int& col , const int& row)=0;
      virtual bool  isDead       ( const uint32_t& detID , const int& col , const int& row)=0;
      virtual bool  isDeadColumn ( const uint32_t& detID , const int& col , const int& row)=0;
      virtual bool  isNoisy       ( const uint32_t& detID , const int& col , const int& row)=0;
      virtual bool  isNoisyColumn ( const uint32_t& detID , const int& col , const int& row)=0;
      virtual void  setESObjects(const edm::EventSetup& es )=0;
      virtual std::vector<uint32_t> getDetIds()=0;
      virtual double getGainLow()=0;
      virtual double getGainHigh()=0;
      virtual double getPedLow()=0;
      virtual double getPedHigh()=0;
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

  virtual bool isNoisy       ( const uint32_t& detID, const int& col, const int& row )=0;
  virtual bool isNoisyColumn ( const uint32_t& detID, const int& col, const int& row )=0;

  void    setESObjects(const edm::EventSetup& es );

  std::vector<uint32_t> getDetIds();
  double getGainLow();
  double getGainHigh();
  double getPedLow();
  double getPedHigh();

 protected:

  float   getPedestalByPixel(const uint32_t& detID,const int& col, const int& row, bool& isDeadPixel, bool& isNoisyPixel) ;
  float   getGainByPixel(const uint32_t& detID,const int& col, const int& row, bool& isDeadPixel, bool& isNoisyPixel) ;

  // the getByColumn functions caches the data to prevent multiple lookups on averaged quanitities
  float   getPedestalByColumn(const uint32_t& detID,const int& col, const int& row, bool& isDeadColumn, bool& isNoisyColumn) ;
  float   getGainByColumn(const uint32_t& detID,const int& col, const int& row, bool& isDeadColumn, bool& isNoisyColumn) ;

  void    throwExepctionForBadRead(std::string payload, const uint32_t& detID, const int& col, const int& row, double value = -1) const;

 private:

  edm::ParameterSet conf_;
  bool ESetupInit_;
  edm::ESHandle<thePayloadObject> ped;
  int numberOfRowsAveragedOver_;
  double gainLow_;
  double gainHigh_;
  double pedLow_;
  double pedHigh_;

  uint32_t old_detID;
  int      old_cols;
  int      old_rocrows;
  // Cache data for payloads that average over columns
  
  // these two quantities determine what column averaged block we are in - i.e. ROC 1 or ROC 2
  int      oldAveragedBlockDataGain_;
  int      oldAveragedBlockDataPed_;
  
  bool     oldThisColumnIsDeadGain_;
  bool     oldThisColumnIsDeadPed_;
  bool     oldThisColumnIsNoisyGain_;
  bool     oldThisColumnIsNoisyPed_;
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
  oldThisColumnIsNoisyGain_ = false;
  oldThisColumnIsNoisyPed_  = false;

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
double SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getGainLow() {
  double gainLow_ = ped->getGainLow();
  return gainLow_;
}

template<class thePayloadObject, class theDBRecordType>
double SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getGainHigh() {
  double gainHigh_ = ped->getGainHigh();
  return gainHigh_;
}

template<class thePayloadObject, class theDBRecordType>
double SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getPedLow() {
  double pedLow_ = ped->getPedLow();
  return pedLow_;
}

template<class thePayloadObject, class theDBRecordType>
double SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getPedHigh() {
  double pedHigh_ = ped->getPedHigh();
  return pedHigh_;
}

template<class thePayloadObject, class theDBRecordType>
float SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getPedestalByPixel(const uint32_t& detID,const int& col, const int& row, bool& isDead, bool& isNoisy) { 
  if(ESetupInit_) {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
      if (detID != old_detID){
	old_detID=detID;
        std::pair<const typename thePayloadObject::Range, const int> rangeAndNCols = ped->getRangeAndNCols(detID, &old_rocrows);
	old_range = rangeAndNCols.first;
	old_cols  = rangeAndNCols.second;
      }
      //std::cout<<" Pedestal "<<ped->getPed(col, row, old_range, old_cols)<<std::endl;
      return  ped->getPed(col, row, old_range, old_cols, isDead, isNoisy, old_rocrows);
  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationServicePayloadGetter::getPedestalByPixel] SiPixelGainCalibrationRcd not initialized ";
}


template<class thePayloadObject, class theDBRecordType>
float SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getGainByPixel(const uint32_t& detID,const int& col, const int& row, bool& isDead, bool& isNoisy) {
  if(ESetupInit_) {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
    if (detID != old_detID){
      old_detID=detID;
      std::pair<const typename thePayloadObject::Range, const int> rangeAndNCols = ped->getRangeAndNCols(detID, &old_rocrows);
      old_range = rangeAndNCols.first;
      old_cols  = rangeAndNCols.second;
    }
    return ped->getGain(col, row, old_range, old_cols, isDead, isNoisy, old_rocrows);
  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationServicePayloadGetter::getGainByPixel] SiPixelGainCalibrationRcd not initialized ";
}


template<class thePayloadObject, class theDBRecordType>
float SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getPedestalByColumn(const uint32_t& detID,const int& col, const int& row, bool& isDeadColumn, bool& isNoisyColumn) { 
  if(ESetupInit_) {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
      // see if we are in the same averaged data block
      bool inTheSameAveragedDataBlock = false;
      if ( row / old_rocrows == oldAveragedBlockDataPed_ )
         inTheSameAveragedDataBlock = true;

      if (detID != old_detID){
	old_detID=detID;
        std::pair<const typename thePayloadObject::Range, const int> rangeAndNCols = ped->getRangeAndNCols(detID, &old_rocrows);
	old_range = rangeAndNCols.first;
	old_cols  = rangeAndNCols.second;
      } 
      else if (col == oldColumnIndexPed_ && inTheSameAveragedDataBlock) // same DetID, same column, same data block
      {
         isDeadColumn = oldThisColumnIsDeadPed_;
         isNoisyColumn = oldThisColumnIsNoisyPed_;
         return oldColumnValuePed_;
      } 

      oldColumnIndexPed_       = col;
      oldAveragedBlockDataPed_ = row / old_rocrows;
      oldColumnValuePed_       = ped->getPed(col, row, old_range, old_cols, isDeadColumn, isNoisyColumn, old_rocrows);
      oldThisColumnIsDeadPed_  = isDeadColumn;
      oldThisColumnIsNoisyPed_  = isNoisyColumn;

      return oldColumnValuePed_;

  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationServicePayloadGetter::getPedestalByColumn] SiPixelGainCalibrationRcd not initialized ";
}


template<class thePayloadObject, class theDBRecordType>
float SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::getGainByColumn(const uint32_t& detID,const int& col, const int& row, bool& isDeadColumn, bool& isNoisyColumn) {
  if(ESetupInit_) {
    //&&&&&&&&&&&&&&&&&&&&
    //Access from DB
    //&&&&&&&&&&&&&&&&&&&&
    bool inTheSameAveragedDataBlock = false;
    if ( row / old_rocrows == oldAveragedBlockDataGain_ )
       inTheSameAveragedDataBlock = true;

    if (detID != old_detID){
      old_detID=detID;
      std::pair<const typename thePayloadObject::Range, const int> rangeAndNCols = ped->getRangeAndNCols(detID, &old_rocrows);
      old_range = rangeAndNCols.first;
      old_cols  = rangeAndNCols.second;
    }
    else if (col == oldColumnIndexGain_ && inTheSameAveragedDataBlock) // same DetID, same column
    {
       isDeadColumn = oldThisColumnIsDeadGain_;
       isDeadColumn = oldThisColumnIsNoisyGain_;
       return oldColumnValueGain_;
    }

    oldColumnIndexGain_       = col;
    oldAveragedBlockDataGain_ = row / old_rocrows;
    oldColumnValueGain_       = ped->getGain(col, row, old_range, old_cols, isDeadColumn, isNoisyColumn, old_rocrows);
    oldThisColumnIsDeadGain_  = isDeadColumn;
    oldThisColumnIsNoisyGain_  = isNoisyColumn;

    return oldColumnValueGain_;

  } else throw cms::Exception("NullPointer")
    << "[SiPixelGainCalibrationServicePayloadGetter::getGainByColumn] SiPixelGainCalibrationRcd not initialized ";
}

template<class thePayloadObject, class theDBRecordType>
void SiPixelGainCalibrationServicePayloadGetter<thePayloadObject,theDBRecordType>::throwExepctionForBadRead(std::string payload, const uint32_t& detID, const int& col, const int& row, const double value) const
{
   std::cerr << "[SiPixelGainCalibrationServicePayloadGetter::SiPixelGainCalibrationServicePayloadGetter]"
      << "[SiPixelGainCalibrationServicePayloadGetter] ERROR - Slow down, speed racer! You have tried to read the ped/gain on a pixel that is flagged as dead/noisy. For payload: " << payload << "  DETID: " 
      << detID << " col: " << col << " row: " << row << ". You must check if the pixel is dead/noisy before asking for the ped/gain value, otherwise you will get corrupt data! value: " << value << std::endl;

   // really yell if this occurs

   edm::LogError("SiPixelGainCalibrationService") << "[SiPixelGainCalibrationServicePayloadGetter::SiPixelGainCalibrationServicePayloadGetter]"
      << "[SiPixelGainCalibrationServicePayloadGetter] ERROR - Slow down, speed racer! You have tried to read the ped/gain on a pixel that is flagged as dead/noisy. For payload: " << payload << "  DETID: " 
      << detID << " col: " << col << " row: " << row << ". You must check if the pixel is dead/noisy before asking for the ped/gain value, otherwise you will get corrupt data! value: " << value << std::endl;
}


#endif
