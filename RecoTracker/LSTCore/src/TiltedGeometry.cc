#include "RecoTracker/LSTCore/interface/TiltedGeometry.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

lst::TiltedGeometry::TiltedGeometry(std::string const& filename) { load(filename); }

void lst::TiltedGeometry::load(std::string const& filename) {
  drdzs_.clear();
  dxdys_.clear();

  std::ifstream ifile(filename, std::ios::binary);
  if (!ifile.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  while (!ifile.eof()) {
    unsigned int detid;
    float drdz, dxdy;

    // Read the detid, drdz, and dxdy from binary file
    ifile.read(reinterpret_cast<char*>(&detid), sizeof(detid));
    ifile.read(reinterpret_cast<char*>(&drdz), sizeof(drdz));
    ifile.read(reinterpret_cast<char*>(&dxdy), sizeof(dxdy));

    if (ifile) {
      drdzs_[detid] = drdz;
      dxdys_[detid] = dxdy;
    } else {
      // End of file or read failed
      if (!ifile.eof()) {
        throw std::runtime_error("Failed to read Tilted Geometry binary data.");
      }
    }
  }
}

float lst::TiltedGeometry::getDrDz(unsigned int detid) const {
  auto res = drdzs_.find(detid);
  return res == drdzs_.end() ? 0.f : res->second;
}

float lst::TiltedGeometry::getDxDy(unsigned int detid) const {
  auto res = dxdys_.find(detid);
  return res == dxdys_.end() ? 0.f : res->second;
}
