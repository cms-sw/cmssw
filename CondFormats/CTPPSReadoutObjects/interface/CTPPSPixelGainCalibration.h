#ifndef CondFormats_CTPPSReadoutObjects_CTPPSPixelGainCalibration_h
#define CondFormats_CTPPSReadoutObjects_CTPPSPixelGainCalibration_h
// -*- C++ -*-
//
// Package:    CTPPSReadoutObjects
// Class:      CTPPSPixelGainCalibration
// 
/**\class CTPPSPixelGainCalibration CTPPSPixelGainCalibration.h CondFormats/CTPPSObjects/src/CTPPSPixelGainCalibration.cc

 Description: Gain calibration object for the CTPPS 3D Pixels.  Store gain/pedestal information at pixel granularity

 Implementation:
     <Notes on implementation>
*/
//  Based on SiPixelObjects/SiPixelGainCalibration 
// Original Author:  Vincenzo Chiochia
//         Created:  Tue 8 12:31:25 CEST 2007
//         Modified: Evan Friis
// $Id: CTPPSPixelGainCalibration.h,v 1.8 2009/02/10 17:25:42 rougny Exp $
//  Adapted for CTPPS : Clemencia Mora Herrera       November 2016
//
#include "CondFormats/Serialization/interface/Serializable.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<vector>

class CTPPSPixelGainCalibration {

 public:
  
  struct DetRegistry{ //to index the channels in each sensor 
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;
    int ncols;
    
    COND_SERIALIZABLE;
  };
  
  
  // Constructors
  CTPPSPixelGainCalibration();
  CTPPSPixelGainCalibration(float minPed, float maxPed, float minGain, float maxGain);
  CTPPSPixelGainCalibration(const uint32_t& detid, const std::vector<float>& peds, const std::vector<float>& gains, 
			    float minPed=0., float maxPed=255., float minGain=0., float maxGain=255.);
  ~CTPPSPixelGainCalibration(){}

  void initialize(){}

  void setGainsPeds(const uint32_t& detId, const std::vector<float>& peds, const std::vector<float>& gains);

  double getGainLow() const { return minGain_; }
  double getGainHigh() const { return maxGain_; }
  double getPedLow() const { return minPed_; }
  double getPedHigh() const { return maxPed_; }

 
  // Set and get public methods

  void putData(uint32_t ipix, float ped, float gain);

  float getPed   (const int& col, const int& row) const;
  float getGain  (const int& col, const int& row) const;
  float getPed   (const uint32_t ipix)const;
  float getGain  (const uint32_t ipix)const;
  int   getNCols () const {return indexes.ncols;}
  int   getIEnd  () const {return indexes.iend;}
  uint32_t getId () const {return indexes.detid;}
  private:

  void resetPixData(uint32_t ipix, float ped, float gain);

  std::vector<float> v_pedestals; 
  std::vector<float> v_gains;
  DetRegistry indexes;

  float  minPed_, maxPed_, minGain_, maxGain_;


 COND_SERIALIZABLE;
};
    
#endif
