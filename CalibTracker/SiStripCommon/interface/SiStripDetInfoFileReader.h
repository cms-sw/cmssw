#ifndef CalibTracker_SiStripChannelGain_SiStripDetInfoFileReader_h
#define CalibTracker_SiStripChannelGain_SiStripDetInfoFileReader_h
// -*- C++ -*-
//
// Package:    SiStripDetInfoFileReader
// Class:      SiStripDetInfoFileReader
//
/**\class SiStripDetInfoFileReader SiStripDetInfoFileReader.cc CalibTracker/SiStripCommon/src/SiStripDetInfoFileReader.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  G. Bruno
//         Created:  Mon Nov 20 10:04:31 CET 2006
//
//

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "CalibFormats/SiStripObjects/interface/SiStripDetInfo.h"
#include <cstdint>

class SiStripDetInfoFileReader {
public:
  using DetInfo = SiStripDetInfo::DetInfo;

  constexpr static char const* const kDefaultFile = "CalibTracker/SiStripCommon/data/SiStripDetInfo.dat";
  explicit SiStripDetInfoFileReader(){};

  explicit SiStripDetInfoFileReader(std::string filePath);
  explicit SiStripDetInfoFileReader(const SiStripDetInfoFileReader&);

  ~SiStripDetInfoFileReader();

  SiStripDetInfoFileReader& operator=(const SiStripDetInfoFileReader& copy);

  SiStripDetInfo const& info() const { return info_; }

  const std::vector<uint32_t>& getAllDetIds() const { return info_.getAllDetIds(); }

  const std::pair<unsigned short, double> getNumberOfApvsAndStripLength(uint32_t detId) const {
    return info_.getNumberOfApvsAndStripLength(detId);
  }

  const float& getThickness(uint32_t detId) const { return info_.getThickness(detId); }

  const std::map<uint32_t, DetInfo>& getAllData() const { return info_.getAllData(); }

private:
  void reader(std::string filePath);

  std::ifstream inputFile_;

  SiStripDetInfo info_;
};
#endif
