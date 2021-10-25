// -*- C++ -*-
// Package:    SiStripCommon
// Class:      SiStripDetInfoFileReader
// Original Author:  G. Bruno
//         Created:  Mon May 20 10:04:31 CET 2007

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <fstream>

SiStripDetInfo SiStripDetInfoFileReader::read(std::string filePath) {
  edm::LogInfo("SiStripDetInfoFileReader") << "filePath " << filePath;

  std::map<uint32_t, DetInfo> detData_;
  std::vector<uint32_t> detIds_;

  std::ifstream inputFile;
  inputFile.open(filePath.c_str());

  if (inputFile.is_open()) {
    for (;;) {
      uint32_t detid;
      double stripLength;
      unsigned short numberOfAPVs;
      float thickness;

      inputFile >> detid >> numberOfAPVs >> stripLength >> thickness;

      if (!(inputFile.eof() || inputFile.fail())) {
        detIds_.push_back(detid);

        //	inputFile >> numberOfAPVs;
        //	inputFile >> stripLength;

        //     	edm::LogInfo("SiStripDetInfoFileReader::SiStripDetInfoFileReader") << detid <<" " <<numberOfAPVs <<" " <<stripLength << " "<< thickness<< endl;

        if (detData_.find(detid) == detData_.end()) {
          detData_[detid] = DetInfo(numberOfAPVs, stripLength, thickness);
        } else {
          edm::LogError("SiStripDetInfoFileReader::SiStripDetInfoFileReader")
              << "DetId " << detid << " already found on file. Ignoring new data";

          detIds_.pop_back();
          continue;
        }
      } else if (inputFile.eof()) {
        edm::LogInfo("SiStripDetInfoFileReader::SiStripDetInfoFileReader - END of file reached");
        break;

      } else if (inputFile.fail()) {
        edm::LogError("SiStripDetInfoFileReader::SiStripDetInfoFileReader - ERROR while reading file");
        break;
      }
    }
    inputFile.close();
  } else {
    edm::LogError("SiStripDetInfoFileReader::SiStripDetInfoFileReader - Unable to open file");
    return SiStripDetInfo();
  }

  return SiStripDetInfo(std::move(detData_), std::move(detIds_));
}
