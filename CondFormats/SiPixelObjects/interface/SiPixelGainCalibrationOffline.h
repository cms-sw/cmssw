#ifndef CondFormats_SiPixelObjects_SiPixelGainCalibrationOffline_h
#define CondFormats_SiPixelObjects_SiPixelGainCalibrationOffline_h
// -*- C++ -*-
//
// Package:    SiPixelObjects
// Class:      SiPixelGainCalibrationOffline
// 
/**\class SiPixelGainCalibrationOffline SiPixelGainCalibrationOffline.h CondFormats/SiPixelObjects/src/SiPixelGainCalibrationOffline.cc

 Description: Gain calibration object for the Silicon Pixel detector.  Stores pedestal at pixel granularity, gain at column granularity.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Modified: Evan Friis
//         Created:  Tue 8 12:31:25 CEST 2007
// $Id: SiPixelGainCalibrationOffline.h,v 1.1 2008/02/06 16:29:55 friis Exp $
//
//
#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>

class SiPixelGainCalibrationOffline {

 public:

  // SiPixelGainCalibrationOffline BLOB FORMAT
  // 
  // For each column, N pedestal values are stored, where N is the number of columns
  // in the CMSSW DetID frame (NOT the ROC frame).  At the end of each N pedestal values,
  // two gain averages are stored, for the two different ROCS.  FOR PLAQUETTES WITH ONLY ONE
  // ROW OF ROCS THE SECOND GAIN AVERAGE MUST STILL BE FILLED SO THE BLOB STREAM IS STILL PARSED
  // CORRECTLY.  The total number of char entries for each column will be (nRows+2)

  struct DecodingStructure{  
    unsigned int datum :8;
  };
  
  struct DetRegistry{
    uint32_t detid;
    uint32_t ibegin;
    uint32_t iend;
    int ncols;
  };
  
  class StrictWeakOrdering{
  public:
    bool operator() (const DetRegistry& p,const uint32_t& i) const {return p.detid < i;}
  };
  
  typedef std::vector<char>::const_iterator                ContainerIterator;  
  typedef std::pair<ContainerIterator, ContainerIterator>  Range;      
  typedef std::vector<DetRegistry>                         Registry;
  typedef Registry::const_iterator                         RegistryIterator;
  
  // Constructors
  SiPixelGainCalibrationOffline();
  SiPixelGainCalibrationOffline(float minPed, float maxPed, float minGain, float maxGain);
  virtual ~SiPixelGainCalibrationOffline(){};

  bool  put(const uint32_t& detID,Range input, const int& nCols);
  const Range getRange(const uint32_t& detID) const;
  void  getDetIds(std::vector<uint32_t>& DetIds_) const;
  const int getNCols(const uint32_t& detID) const;
  const std::pair<const Range, const int> getRangeAndNCols(const uint32_t& detID) const;

  // Set and get public methods
  void  setDataGain(float gainLow, float gainHigh, const int& nRows, std::vector<char>& vped);
  void  setDataPedestal(float pedestal, std::vector<char>& vped);
  float getPed   (const int& col, const int& row, const Range& range, const int& nCols) const;
  float getGain  (const int& col, const int& row, const Range& range, const int& nCols) const;

  private:

  float   encodeGain(const float& gain);
  float   encodePed (const float& ped);
  float   decodeGain(unsigned int gain) const;
  float   decodePed (unsigned int ped) const;

  std::vector<char> v_pedestals; //@@@ blob streaming doesn't work with uint16_t and with classes
  std::vector<DetRegistry> indexes;
  float  minPed_, maxPed_, minGain_, maxGain_;

};
    
#endif
