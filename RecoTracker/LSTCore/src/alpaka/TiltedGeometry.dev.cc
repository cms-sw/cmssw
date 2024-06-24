#include "TiltedGeometry.h"

SDL::TiltedGeometry<SDL::Dev>::TiltedGeometry(std::string filename) { load(filename); }

void SDL::TiltedGeometry<SDL::Dev>::load(std::string filename) {
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

float SDL::TiltedGeometry<SDL::Dev>::getDrDz(unsigned int detid) const {
  if (drdzs_.find(detid) != drdzs_.end()) {
    return drdzs_.at(detid);
  } else {
    return 0;
  }
}

float SDL::TiltedGeometry<SDL::Dev>::getDxDy(unsigned int detid) const {
  if (dxdys_.find(detid) != dxdys_.end()) {
    return dxdys_.at(detid);
  } else {
    return 0;
  }
}
