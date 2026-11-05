// -*- C++ -*-
//
// Package:    DataFormats
// Class:      testHGCSiliconDetId
//
/**\class DataFormats testHGCSiliconDetId.cc
 test/testHGCSiliconDetId.cc

 Description: <one line file summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2026/09/03
//
//

// system include files
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

std::vector<std::string> splitString(const std::string& fLine) {
  std::vector<std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size(); i++) {
    if (fLine[i] == ' ' || i == fLine.size()) {
      if (!empty) {
        std::string item(fLine, start, i - start);
        result.emplace_back(item);
        empty = true;
      }
      start = i + 1;
    } else {
      if (empty)
        empty = false;
    }
  }
  return result;
}

int main() {
  std::string fname("D120.txt");
  std::cout << "Test change wafer u, v coorinates for HGCSiliconDetId with inputs from " << fname << std::endl
            << std::endl;

  if (!fname.empty()) {
    edm::FileInPath filetmp("DataFormats/ForwardDetId/data/" + fname);
    std::string fileName = filetmp.fullPath();
    std::ifstream fInput(fileName.c_str());
    if (!fInput.good()) {
      std::cout << "Cannot open file " << fileName << std::endl;
    } else {
      char buffer[80];
      while (fInput.getline(buffer, 80)) {
        std::vector<std::string> items = splitString(std::string(buffer));
        if (items.size() == 9) {
          DetId::Detector det = static_cast<DetId::Detector>(std::atoi(items[0].c_str()));
          int32_t type = std::atoi(items[1].c_str());
          int32_t layer = std::atoi(items[2].c_str());
          int32_t waferU = std::atoi(items[3].c_str());
          int32_t waferV = std::atoi(items[4].c_str());
          int32_t cellU = std::atoi(items[5].c_str());
          int32_t cellV = std::atoi(items[6].c_str());
          HGCSiliconDetId id1(det, 1, type, layer, waferU, waferV, cellU, cellV);
          int32_t waferNU = std::atoi(items[7].c_str());
          int32_t waferNV = std::atoi(items[8].c_str());
          uint32_t id2 = HGCSiliconDetId::waferUVset(id1.rawId(), waferNU, waferNV);
          std::cout << "Modify wafer coordinates of " << id1 << " with (" << waferNU << ", " << waferNV << ") to get "
                    << HGCSiliconDetId(id2) << std::endl;
        }
      }
      fInput.close();
    }
  }
}
