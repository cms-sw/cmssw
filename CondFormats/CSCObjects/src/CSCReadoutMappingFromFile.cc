#include <CondFormats/CSCObjects/interface/CSCReadoutMappingFromFile.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/ParameterSet/interface/FileInPath.h>
#include <iostream>
#include <fstream>
#include <sstream>

CSCReadoutMappingFromFile::CSCReadoutMappingFromFile(const edm::ParameterSet& ps) { fill(ps); }

CSCReadoutMappingFromFile::~CSCReadoutMappingFromFile() {}

void CSCReadoutMappingFromFile::fill(const edm::ParameterSet& ps) {
  edm::FileInPath fp = ps.getParameter<edm::FileInPath>("theMappingFile");
  theMappingFile = fp.fullPath();
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
