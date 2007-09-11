//----------------------------------------------------------------------------
//! \class PixelGainCalibration
//! \brief A C++ container for the pixel gain calibration. Modified version
//!        of the strip container.
//!
//! \author Vincenzo Chiochia
//!
//----------------------------------------------------------------------------

#ifndef SiPixelGainCalibration_h
#define SiPixelGainCalibration_h

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
  virtual ~SiPixelGainCalibration(){};

  bool  put(const uint32_t& detID,Range input, const int& nCols);
  const Range getRange(const uint32_t& detID) const;
  void  getDetIds(std::vector<uint32_t>& DetIds_) const;
  const int getNCols(const uint32_t& detID) const;

  void  setData(float ped, float gain, std::vector<char>& vped);
  float getPed   (const int& col, const int& row, const Range& range, const int& nCols) const;
  float getGain  (const int& col, const int& row, const Range& range, const int& nCols) const;
  void  setPedRange(float minPed, float maxPed);
  void  setGainRange(float minGain, float maxGain);

  private:

  float   encodeGain(const float& gain);
  float   encodePed (const float& ped);
  float   decodeGain(unsigned int gain) const;
  float   decodePed (unsigned int ped) const;

  std::vector<char> v_pedestals; //@@@ blob streaming doesn't work with uint16_t and with classes
  std::vector<DetRegistry> indexes;
  float  minGain_, maxGain_, minPed_, maxPed_;

};
    
#endif
