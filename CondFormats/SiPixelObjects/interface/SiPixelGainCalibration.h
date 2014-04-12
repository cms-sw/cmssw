#ifndef CondFormats_SiPixelObjects_SiPixelGainCalibration_h
#define CondFormats_SiPixelObjects_SiPixelGainCalibration_h
// -*- C++ -*-
//
// Package:    SiPixelObjects
// Class:      SiPixelGainCalibration
// 
/**\class SiPixelGainCalibration SiPixelGainCalibration.h CondFormats/SiPixelObjects/src/SiPixelGainCalibration.cc

 Description: Gain calibration object for the Silicon Pixel detector.  Store gain/pedestal information at pixel granularity

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia
//         Created:  Tue 8 12:31:25 CEST 2007
//         Modified: Evan Friis
// $Id: SiPixelGainCalibration.h,v 1.8 2009/02/10 17:25:42 rougny Exp $
//
//
#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>

class SiPixelGainCalibration {

 public:

  struct DecodingStructure{  
    unsigned int gain :8;
    unsigned int ped  :8;
    //    unsigned int ped :10;
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
  SiPixelGainCalibration();
  SiPixelGainCalibration(float minPed, float maxPed, float minGain, float maxGain);
  ~SiPixelGainCalibration(){}

  void initialize(){}

  bool  put(const uint32_t& detID,Range input, const int& nCols);
  const Range getRange(const uint32_t& detID) const;
  void  getDetIds(std::vector<uint32_t>& DetIds_) const;
  const int getNCols(const uint32_t& detID) const;
  const std::pair<const Range, const int> getRangeAndNCols(const uint32_t& detID) const;

  unsigned int getNumberOfRowsToAverageOver() const { return numberOfRowsToAverageOver_; }
  double getGainLow() const { return minGain_; }
  double getGainHigh() const { return maxGain_; }
  double getPedLow() const { return minPed_; }
  double getPedHigh() const { return maxPed_; }

  // Set and get public methods
  void  setData(float ped, float gain, std::vector<char>& vped, bool thisPixelIsDead = false, bool thisPixelIsNoisy = false);

  void  setDeadPixel(std::vector<char>& vped)  { setData(0, 0, /*dummy values, not used*/ vped,  true , false ); }
  void  setNoisyPixel(std::vector<char>& vped) { setData(0, 0, /*dummy values, not used*/ vped,  false , true ); }

  float getPed   (const int& col, const int& row, const Range& range, const int& nCols, bool& isDead, bool& isNoisy) const;
  float getGain  (const int& col, const int& row, const Range& range, const int& nCols, bool& isDead, bool& isNoisy) const;

  private:

  float   encodeGain(const float& gain);
  float   encodePed (const float& ped);
  float   decodeGain(unsigned int gain) const;
  float   decodePed (unsigned int ped) const;

  std::vector<char> v_pedestals; //@@@ blob streaming doesn't work with uint16_t and with classes
  std::vector<DetRegistry> indexes;
  float  minPed_, maxPed_, minGain_, maxGain_;

  unsigned int numberOfRowsToAverageOver_;   //THIS WILL BE HARDCODED TO 1 (no averaging) DON'T CHANGE UNLESS YOU KNOW WHAT YOU ARE DOING! 
  unsigned int nBinsToUseForEncoding_;
  unsigned int deadFlag_;
  unsigned int noisyFlag_;

};
    
#endif
