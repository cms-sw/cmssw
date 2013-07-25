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
// $Id: SiStripDetInfoFileReader.h,v 1.4 2008/09/19 16:53:10 giordano Exp $
//
//

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/cstdint.hpp>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripDetInfoFileReader  {

public:

  struct DetInfo{
    
    DetInfo(){};
    DetInfo(unsigned short _nApvs, double _stripLength, float _thickness):
    nApvs(_nApvs),stripLength(_stripLength),thickness(_thickness){};

    unsigned short nApvs;
    double stripLength;
    float thickness;
  };

  explicit SiStripDetInfoFileReader(){};
  explicit SiStripDetInfoFileReader(const edm::ParameterSet&,
				    const edm::ActivityRegistry&);

  explicit SiStripDetInfoFileReader(std::string filePath);
  explicit SiStripDetInfoFileReader(const SiStripDetInfoFileReader&);

  ~SiStripDetInfoFileReader();

  SiStripDetInfoFileReader& operator=(const SiStripDetInfoFileReader &copy);

  const std::vector<uint32_t> & getAllDetIds() const {return detIds_;}

  const std::pair<unsigned short, double>  getNumberOfApvsAndStripLength(uint32_t detId) const;

  const float & getThickness(uint32_t detId) const;

  const std::map<uint32_t, DetInfo > & getAllData() const {return detData_;}


private:

  void reader(std::string filePath);

  std::ifstream inputFile_; 
  //  std::string filePath_;

  std::map<uint32_t, DetInfo> detData_;
  //  std::map<uint32_t, std::pair<unsigned short, double> > detData_;
  //std::map<uint32_t, float > detThickness_;
  std::vector<uint32_t> detIds_;

};
#endif
