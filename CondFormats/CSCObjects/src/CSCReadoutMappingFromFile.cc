#include "CondFormats/CSCObjects/interface/CSCReadoutMappingFromFile.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <fstream>
#include <sstream>

CSCReadoutMappingFromFile::CSCReadoutMappingFromFile(std::string iName) { fill(std::move(iName)); }

CSCReadoutMappingFromFile::~CSCReadoutMappingFromFile() {}

void CSCReadoutMappingFromFile::fill(std::string fileName) {
  theMappingFile = std::move(fileName);
  std::ifstream in(theMappingFile.c_str());
  std::string line;
  const std::string commentFlag = "#";
  if (!in) {
    edm::LogError("CSC") << " Failed to open file " << theMappingFile << " containing mapping.";
  } else {
    edm::LogInfo("CSC") << " Opened file " << theMappingFile << " containing mapping.";

    while (getline(in, line)) {  // getline() from <string>
      // LogDebug("CSC") << line;
      if (line[0] != commentFlag[0]) {
        int i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11;
        std::istringstream is(line);
        is >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7 >> i8 >> i9 >> i10 >> i11;
        // LogDebug("CSC") << i1 << " " << i2 << " " << i3 << " " << i4 << " " <<
        //	  i5 << " " << i6 << " " << i7 << " " << i8 << " " << i9;
        addRecord(i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11);
      }
    }
  }

  return;
}
