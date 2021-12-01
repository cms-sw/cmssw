#ifndef L1TMuonOverlapFwVersion_h
#define L1TMuonOverlapFwVersion_h

#include <memory>
#include <iostream>
#include <vector>
#include <cmath>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/L1TObjects/interface/LUT.h"

///////////////////////////////////////
///////////////////////////////////////
class L1TMuonOverlapFwVersion {
public:
  L1TMuonOverlapFwVersion() {
    algorithmVer_ = 0x110;
    layersVer_ = 0x6;
    patternsVer_ = 0x3;
    synthDate_ = "2018-9-18 21:26:2";
  }
  L1TMuonOverlapFwVersion(unsigned algoV, unsigned layersV, unsigned patternsV, std::string sDate) {
    algorithmVer_ = algoV;
    layersVer_ = layersV;
    patternsVer_ = patternsV;
    synthDate_ = sDate;
  }
  ~L1TMuonOverlapFwVersion() {}

  unsigned algoVersion() const { return algorithmVer_; }
  unsigned layersVersion() const { return layersVer_; }
  unsigned fwVersion() const { return layersVer_; }
  unsigned patternsVersion() const { return patternsVer_; }
  std::string synthDate() const { return synthDate_; }
  void setAlgoVersion(unsigned algoV) { algorithmVer_ = algoV; }
  void setLayersVersion(unsigned layersV) { layersVer_ = layersV; }
  void setFwVersion(unsigned layersV) { layersVer_ = layersV; }
  void setPatternsVersion(unsigned patternsV) { patternsVer_ = patternsV; }
  void setSynthDate(std::string sDate) { synthDate_ = sDate; }

  ///Firmware configuration parameters
  unsigned algorithmVer_;
  unsigned layersVer_;
  unsigned patternsVer_;
  std::string synthDate_;

  COND_SERIALIZABLE;
};
#endif
