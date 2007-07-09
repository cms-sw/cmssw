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
// $Id: SiStripDetInfoFileReader.h,v 1.1 2007/06/13 14:03:35 gbruno Exp $
//
//

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/cstdint.hpp>

class SiStripDetInfoFileReader  {

public:

  explicit SiStripDetInfoFileReader(std::string filePath);
  ~SiStripDetInfoFileReader();


  const std::vector<uint32_t> & getAllDetIds() const;

  const std::pair<unsigned short, double> & getNumberOfApvsAndStripLength(uint32_t detId) const;

  const float & getThickness(uint32_t detId) const;



private:


  std::ifstream inputFile_; 
  //  std::string filePath_;

  std::map<uint32_t, std::pair<unsigned short, double> > detData_;
  std::map<uint32_t, float > detThickness_;
  std::vector<uint32_t> detIds_;

};
#endif
