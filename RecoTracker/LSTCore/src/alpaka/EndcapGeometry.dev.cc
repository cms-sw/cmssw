#include "EndcapGeometry.h"

SDL::EndcapGeometry<SDL::Dev>::EndcapGeometry(SDL::Dev const& devAccIn,
                                              SDL::QueueAcc& queue,
                                              SDL::EndcapGeometryHost<SDL::Dev> const& endcapGeometryIn)
    : geoMapDetId_buf(allocBufWrapper<unsigned int>(devAccIn, endcapGeometryIn.centroid_phis_.size())),
      geoMapPhi_buf(allocBufWrapper<float>(devAccIn, endcapGeometryIn.centroid_phis_.size())) {
  dxdy_slope_ = endcapGeometryIn.dxdy_slope_;
  centroid_phis_ = endcapGeometryIn.centroid_phis_;
  fillGeoMapArraysExplicit(queue);
}

void SDL::EndcapGeometryHost<SDL::Dev>::load(std::string filename) {
  dxdy_slope_.clear();
  centroid_phis_.clear();

  std::ifstream ifile(filename, std::ios::binary);
  if (!ifile.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  while (!ifile.eof()) {
    unsigned int detid;
    float dxdy_slope, centroid_phi;

    // Read the detid, dxdy_slope, and centroid_phi from binary file
    ifile.read(reinterpret_cast<char*>(&detid), sizeof(detid));
    ifile.read(reinterpret_cast<char*>(&dxdy_slope), sizeof(dxdy_slope));
    ifile.read(reinterpret_cast<char*>(&centroid_phi), sizeof(centroid_phi));

    if (ifile) {
      dxdy_slope_[detid] = dxdy_slope;
      centroid_phis_[detid] = centroid_phi;
    } else {
      // End of file or read failed
      if (!ifile.eof()) {
        throw std::runtime_error("Failed to read Endcap Geometry binary data.");
      }
    }
  }
}

void SDL::EndcapGeometry<SDL::Dev>::fillGeoMapArraysExplicit(SDL::QueueAcc& queue) {
  unsigned int phi_size = centroid_phis_.size();

  // Allocate buffers on host
  SDL::DevHost const& devHost = cms::alpakatools::host();
  auto mapPhi_host_buf = allocBufWrapper<float>(devHost, phi_size);
  auto mapDetId_host_buf = allocBufWrapper<unsigned int>(devHost, phi_size);

  // Access the raw pointers of the buffers
  float* mapPhi = alpaka::getPtrNative(mapPhi_host_buf);
  unsigned int* mapDetId = alpaka::getPtrNative(mapDetId_host_buf);

  unsigned int counter = 0;
  for (auto it = centroid_phis_.begin(); it != centroid_phis_.end(); ++it) {
    unsigned int detId = it->first;
    float Phi = it->second;
    mapPhi[counter] = Phi;
    mapDetId[counter] = detId;
    counter++;
  }

  nEndCapMap = counter;

  // Copy data from host to device buffers
  alpaka::memcpy(queue, geoMapPhi_buf, mapPhi_host_buf);
  alpaka::memcpy(queue, geoMapDetId_buf, mapDetId_host_buf);
  alpaka::wait(queue);
}

float SDL::EndcapGeometry<SDL::Dev>::getdxdy_slope(unsigned int detid) const {
  if (dxdy_slope_.find(detid) != dxdy_slope_.end()) {
    return dxdy_slope_.at(detid);
  } else {
    return 0;
  }
}
float SDL::EndcapGeometryHost<SDL::Dev>::getdxdy_slope(unsigned int detid) const {
  if (dxdy_slope_.find(detid) != dxdy_slope_.end()) {
    return dxdy_slope_.at(detid);
  } else {
    return 0;
  }
}
