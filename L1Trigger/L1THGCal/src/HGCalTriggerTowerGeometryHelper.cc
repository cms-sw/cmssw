
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTowerGeometryHelper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <iostream>
#include <fstream>


HGCalTriggerTowerGeometryHelper::HGCalTriggerTowerGeometryHelper(const edm::ParameterSet& conf) : minEta(conf.getParameter<double>("minEta")),
                                                                                                  maxEta(conf.getParameter<double>("maxEta")),
                                                                                                  minPhi(conf.getParameter<double>("minPhi")),
                                                                                                  maxPhi(conf.getParameter<double>("maxPhi")),
                                                                                                  nBinsEta(conf.getParameter<int>("nBinsEta")),
                                                                                                  nBinsPhi(conf.getParameter<int>("nBinsPhi")),
                                                                                                  binsEta(conf.getParameter<std::vector<double> >("binsEta")),
                                                                                                  binsPhi(conf.getParameter<std::vector<double> >("binsPhi")) {


  if(!binsEta.empty() && ((unsigned int)(binsEta.size()) != nBinsEta+1)) {
    throw edm::Exception(edm::errors::Configuration, "Configuration")
      << "HGCalTriggerTowerGeometryHelper nBinsEta for the tower map not consistent with binsEta size"<<std::endl;
  }

  if(!binsPhi.empty() && ((unsigned int)(binsPhi.size()) != nBinsPhi+1)) {
    throw edm::Exception(edm::errors::Configuration, "Configuration")
      << "HGCalTriggerTowerGeometryHelper nBinsPhi for the tower map not consistent with binsPhi size"<<std::endl;
  }

  // if the bin vecctor is empty we assume the bins to be regularly spaced
  if(binsEta.empty()) {
    for(unsigned int bin1 = 0; bin1 != nBinsEta+1; bin1++) {
      binsEta.push_back(minEta+bin1*((maxEta-minEta)/nBinsEta));
    }
  }

  // if the bin vecctor is empty we assume the bins to be regularly spaced
  if(binsPhi.empty()) {
    for(unsigned int bin2 = 0; bin2 != nBinsPhi+1; bin2++) {
      binsPhi.push_back(minPhi+bin2*((maxPhi-minPhi)/nBinsPhi));
    }
  }

  for(int zside = -1; zside <= 1; zside+=2) {
    for(unsigned int bin1 = 0; bin1 != nBinsEta; bin1++) {
      for(unsigned int bin2 = 0; bin2 != nBinsPhi; bin2++) {
        l1t::HGCalTowerID towerId(zside, bin1, bin2);
        tower_coords_.emplace_back(towerId.rawId(),
                                   (binsEta[bin1+1] + binsEta[bin1])/2,
                                   (binsPhi[bin2+1] + binsPhi[bin2])/2);
      }
    }
  }

  if(conf.getParameter<bool>("readMappingFile")) {
    // We read the TC to TT mapping from file,
    // otherwise we derive the TC to TT mapping on the fly from eta-phi coord. of the TCs
    std::ifstream l1tTriggerTowerMappingStream(conf.getParameter<edm::FileInPath>("L1TTriggerTowerMapping").fullPath());
    if(!l1tTriggerTowerMappingStream.is_open()) {
        throw cms::Exception("MissingDataFile")
            << "Cannot open HGCalTriggerGeometry L1TTriggerTowerMapping file\n";
    }

    unsigned trigger_cell_id = 0;
    unsigned short ix = 0;
    unsigned short iy = 0;

    for(; l1tTriggerTowerMappingStream >> trigger_cell_id >> ix >> iy;) {
      HGCalDetId detId(trigger_cell_id);
      int zside = detId.zside();
      l1t::HGCalTowerID towerId(zside, ix, iy);
      cells_to_trigger_towers_[trigger_cell_id] = towerId.rawId();
    }
    l1tTriggerTowerMappingStream.close();

  }
}


const std::vector<l1t::HGCalTowerCoord>& HGCalTriggerTowerGeometryHelper::getTowerCoordinates() const {
  return tower_coords_;
}


int HGCalTriggerTowerGeometryHelper::binBinarySearch(const std::vector<double>& vec, int start, int end, const double& key) const {
    // Termination condition: start index greater than end index
    if(start > end) {
      return -1;
    }
    // Find the middle element of the vector and use that for splitting
    // the array into two pieces.
    const int middle = start + ((end - start) / 2);
    if(vec[middle] <= key && key < vec[middle+1]) {
        return middle;
    } else if(vec[middle] > key) {
        return binBinarySearch(vec, start, middle, key);
    }
    return binBinarySearch(vec, middle + 1, end, key);
};


unsigned short HGCalTriggerTowerGeometryHelper::getTriggerTowerFromTriggerCell(const unsigned trigger_cell_id, const float& eta, const float& phi) const {
  // NOTE: if the TC is not found in the map than it is mapped via eta-phi coords.
  // this can be considered dangerous (silent failure of the map) but it actually allows to save
  // memory mapping explicitly only what is actually needed
  auto tower_id_itr = cells_to_trigger_towers_.find(trigger_cell_id);
  if(tower_id_itr != cells_to_trigger_towers_.end()) return tower_id_itr->second;

  int bin_eta = binBinarySearch(binsEta, 0, nBinsEta, fabs(eta));
  int bin_phi = binBinarySearch(binsPhi, 0, nBinsPhi, phi);
  // we add a protection for TCs in Hadron part which are outside the boundaries and possible rounding effects
  if(bin_eta == -1) {
    if(fabs(eta) < minEta) {
      bin_eta = 0;
    } else if(fabs(eta) >= maxEta) {
      bin_eta = nBinsEta;
    } else {
      edm::LogError("HGCalTriggerTowerGeometryHelper") << " did not manage to map TC " << trigger_cell_id << " (eta: " << eta << ") to any Trigger Tower\n";
    }
  }
  if(bin_phi == -1) {
    if(phi < minPhi) {
      bin_phi = nBinsPhi;
    } else if(phi >= maxPhi) {
      bin_phi = 0;
    } else {
      edm::LogError("HGCalTriggerTowerGeometryHelper") << " did not manage to map TC " << trigger_cell_id << " (phi: " << phi << ") to any Trigger Tower\n";
    }
  }
  int zside = eta < 0 ?  -1 : 1;
  // std::cout << "TC " << trigger_cell_id << " eta: " << eta << " phi: " << phi << " mapped to bin: (" << zside << ", " << bin_eta << ", " << bin_phi << ")" << std::endl;
  return l1t::HGCalTowerID(zside, bin_eta, bin_phi).rawId();
}
