// -*- C++ -*-
// Package:    SiStripCommon
// Class:      SiStripDetInfoFileReader
// Original Author:  G. Bruno
//         Created:  Mon May 20 10:04:31 CET 2007

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

using namespace cms;
using namespace std;

SiStripDetInfoFileReader& SiStripDetInfoFileReader::operator=(const SiStripDetInfoFileReader& copy) {
  info_ = copy.info_;
  return *this;
}

SiStripDetInfoFileReader::SiStripDetInfoFileReader(const SiStripDetInfoFileReader& copy) : info_{copy.info_} {}

SiStripDetInfoFileReader::SiStripDetInfoFileReader(std::string filePath) { reader(filePath); }

void SiStripDetInfoFileReader::reader(std::string filePath) {
  //   if(filePath==std::string("")){
  //     filePath = edm::FileInPath(std::string("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat") ).fullPath();
  //   }

  edm::LogInfo("SiStripDetInfoFileReader") << "filePath " << filePath << std::endl;

  std::map<uint32_t, DetInfo> detData_;
  std::vector<uint32_t> detIds_;

  inputFile_.open(filePath.c_str());

  if (inputFile_.is_open()) {
    for (;;) {
      uint32_t detid;
      double stripLength;
      unsigned short numberOfAPVs;
      float thickness;

      inputFile_ >> detid >> numberOfAPVs >> stripLength >> thickness;

      if (!(inputFile_.eof() || inputFile_.fail())) {
        detIds_.push_back(detid);

        //	inputFile_ >> numberOfAPVs;
        //	inputFile_ >> stripLength;

        //     	edm::LogInfo("SiStripDetInfoFileReader::SiStripDetInfoFileReader") << detid <<" " <<numberOfAPVs <<" " <<stripLength << " "<< thickness<< endl;

        std::map<uint32_t, DetInfo>::const_iterator it = detData_.find(detid);

        if (it == detData_.end()) {
          detData_[detid] = DetInfo(numberOfAPVs, stripLength, thickness);
        } else {
          edm::LogError("SiStripDetInfoFileReader::SiStripDetInfoFileReader")
              << "DetId " << detid << " already found on file. Ignoring new data" << endl;

          detIds_.pop_back();
          continue;
        }
      } else if (inputFile_.eof()) {
        edm::LogInfo("SiStripDetInfoFileReader::SiStripDetInfoFileReader - END of file reached") << endl;
        break;

      } else if (inputFile_.fail()) {
        edm::LogError("SiStripDetInfoFileReader::SiStripDetInfoFileReader - ERROR while reading file") << endl;
        break;
      }
    }

    inputFile_.close();

  } else {
    edm::LogError("SiStripDetInfoFileReader::SiStripDetInfoFileReader - Unable to open file") << endl;
    return;
  }

  info_ = SiStripDetInfo(detData_, detIds_);
  //   int i=0;
  //   for(std::map<uint32_t, std::pair<unsigned short, double> >::iterator it =detData_.begin(); it!=detData_.end(); it++ ) {
  //     std::cout<< it->first << " " << (it->second).first << " " << (it->second).second<<endl;
  //     i++;
  //   }
  //   std::cout<<i;
}

SiStripDetInfoFileReader::~SiStripDetInfoFileReader() {}
