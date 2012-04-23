#ifndef CalibTracker_SiPixelESProducers_SiPixelDetInfoFileReader_h
#define CalibTracker_SiPixelESProducers_SiPixelDetInfoFileReader_h
// -*- C++ -*-
//
// Package:    SiPixelDetInfoFileReader
// Class:      SiPixelDetInfoFileReader
// 
/**\class SiPixelDetInfoFileReader SiPixelDetInfoFileReader.cc CalibTracker/SiPixelCommon/src/SiPixelDetInfoFileReader.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  V.Chiochia
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id: SiPixelDetInfoFileReader.h,v 1.1.2.1 2010/06/25 16:41:53 hcheung Exp $
//
//

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/cstdint.hpp>

class SiPixelDetInfoFileReader  {

public:

  explicit SiPixelDetInfoFileReader(std::string filePath, bool readROCInfo=false);
  ~SiPixelDetInfoFileReader();

  const std::vector<uint32_t> & getAllDetIds() const;
  const std::pair<int, int> & getDetUnitDimensions(uint32_t detId) const;
  const int & getROCRows(uint32_t detId) const;

private:

  std::ifstream inputFile_; 
  //  std::string filePath_;

  std::map<uint32_t, std::pair<int, int> > detData_;
  std::vector<uint32_t> detIds_;
  std::map<uint32_t, int>  theROCInfo_;

};
#endif
