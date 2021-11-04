
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTowerGeometryHelper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>

HGCalTriggerTowerGeometryHelper::HGCalTriggerTowerGeometryHelper(const edm::ParameterSet& conf)
    : doNose_(conf.getParameter<bool>("doNose")),
      minEta_(conf.getParameter<double>("minEta")),
      maxEta_(conf.getParameter<double>("maxEta")),
      minPhi_(conf.getParameter<double>("minPhi")),
      maxPhi_(conf.getParameter<double>("maxPhi")),
      nBinsEta_(conf.getParameter<int>("nBinsEta")),
      nBinsPhi_(conf.getParameter<int>("nBinsPhi")),
      binsEta_(conf.getParameter<std::vector<double> >("binsEta")),
      binsPhi_(conf.getParameter<std::vector<double> >("binsPhi")),
      splitModuleSum_(conf.getParameter<bool>("splitModuleSum")) {
  if (!binsEta_.empty() && ((unsigned int)(binsEta_.size()) != nBinsEta_ + 1)) {
    throw edm::Exception(edm::errors::Configuration, "Configuration")
        << "HGCalTriggerTowerGeometryHelper nBinsEta for the tower map not consistent with binsEta size" << std::endl;
  }

  if (!binsPhi_.empty() && ((unsigned int)(binsPhi_.size()) != nBinsPhi_ + 1)) {
    throw edm::Exception(edm::errors::Configuration, "Configuration")
        << "HGCalTriggerTowerGeometryHelper nBinsPhi for the tower map not consistent with binsPhi size" << std::endl;
  }

  // if the bin vecctor is empty we assume the bins to be regularly spaced
  if (binsEta_.empty()) {
    for (unsigned int bin1 = 0; bin1 != nBinsEta_ + 1; bin1++) {
      binsEta_.push_back(minEta_ + bin1 * ((maxEta_ - minEta_) / nBinsEta_));
    }
  }

  // if the bin vecctor is empty we assume the bins to be regularly spaced
  if (binsPhi_.empty()) {
    for (unsigned int bin2 = 0; bin2 != nBinsPhi_ + 1; bin2++) {
      binsPhi_.push_back(minPhi_ + bin2 * ((maxPhi_ - minPhi_) / nBinsPhi_));
    }
  }

  for (int zside = -1; zside <= 1; zside += 2) {
    for (unsigned int bin1 = 0; bin1 != nBinsEta_; bin1++) {
      for (unsigned int bin2 = 0; bin2 != nBinsPhi_; bin2++) {
        l1t::HGCalTowerID towerId(doNose_, zside, bin1, bin2);
        tower_coords_.emplace_back(towerId.rawId(),
                                   zside * ((binsEta_[bin1 + 1] + binsEta_[bin1]) / 2),
                                   (binsPhi_[bin2 + 1] + binsPhi_[bin2]) / 2);
      }
    }
  }

  if (conf.getParameter<bool>("readMappingFile")) {
    // We read the TC to TT mapping from file,
    // otherwise we derive the TC to TT mapping on the fly from eta-phi coord. of the TCs
    std::ifstream l1tTriggerTowerMappingStream(conf.getParameter<edm::FileInPath>("L1TTriggerTowerMapping").fullPath());
    if (!l1tTriggerTowerMappingStream.is_open()) {
      throw cms::Exception("MissingDataFile") << "Cannot open HGCalTriggerGeometry L1TTriggerTowerMapping file\n";
    }

    unsigned trigger_cell_id = 0;
    unsigned short iEta = 0;
    unsigned short iPhi = 0;

    for (; l1tTriggerTowerMappingStream >> trigger_cell_id >> iEta >> iPhi;) {
      if (iEta >= nBinsEta_ || iPhi >= nBinsPhi_) {
        throw edm::Exception(edm::errors::Configuration, "Configuration")
            << "HGCalTriggerTowerGeometryHelper warning inconsistent mapping TC : " << trigger_cell_id
            << " to TT iEta: " << iEta << " iPhi: " << iPhi << " when max #bins eta: " << nBinsEta_
            << " phi: " << nBinsPhi_ << std::endl;
      }
      l1t::HGCalTowerID towerId(doNose_, triggerTools_.zside(DetId(trigger_cell_id)), iEta, iPhi);
      cells_to_trigger_towers_[trigger_cell_id] = towerId.rawId();
    }
    l1tTriggerTowerMappingStream.close();
  }

  if (splitModuleSum_) {
    //variables for transforming towers
    rotate180Deg_ = int(nBinsPhi_) / 2;
    rotate120Deg_ = int(nBinsPhi_) / 3;
    reverseX_ = int(nBinsPhi_) / 2 - 1;

    std::ifstream moduleTowerMappingStream(conf.getParameter<edm::FileInPath>("moduleTowerMapping").fullPath());
    if (!moduleTowerMappingStream.is_open()) {
      throw cms::Exception("MissingDataFile") << "Cannot open HGCalTowerMapProducer moduleTowerMapping file\n";
    }
    //get split divisors
    std::string line;
    std::getline(moduleTowerMappingStream, line);  //Skip row
    std::getline(moduleTowerMappingStream, line);
    std::stringstream ss(line);
    ss >> splitDivisorSilic_ >> splitDivisorScint_;

    //get towers and module sum shares
    std::getline(moduleTowerMappingStream, line);  //Skip row
    std::getline(moduleTowerMappingStream, line);  //Skip row
    const int minNumOfWordsPerRow = 5;
    const int numOfWordsPerTower = 3;
    for (std::string line; std::getline(moduleTowerMappingStream, line);) {
      int numOfWordsInThisRow = 0;
      for (std::string::size_type i = 0; i < line.size(); i++) {
        if (line[i] != ' ' && line[i + 1] == ' ') {
          numOfWordsInThisRow++;
        }
      }
      if (numOfWordsInThisRow < minNumOfWordsPerRow) {
        throw edm::Exception(edm::errors::Configuration, "Configuration")
            << "HGCalTriggerTowerGeometryHelper warning: Incorrect/incomplete values for module ID in the mapping "
               "file.\n"
            << "The incorrect line is:" << line << std::endl;
      }
      int subdet;
      int layer;
      int moduleU;
      int moduleV;
      int numTowers;
      std::stringstream ss(line);
      ss >> subdet >> layer >> moduleU >> moduleV >> numTowers;
      if (numOfWordsInThisRow != (numTowers * numOfWordsPerTower + minNumOfWordsPerRow)) {
        throw edm::Exception(edm::errors::Configuration, "Configuration")
            << "HGCalTriggerTowerGeometryHelper warning: Incorrect/incomplete values for module ID or tower "
               "share/eta/phi in the mapping file.\n"
            << "The incorrect line is:" << line << std::endl;
      }
      unsigned packed_modID = packLayerSubdetWaferId(subdet, layer, moduleU, moduleV);
      std::vector<unsigned> towers;
      for (int itr_tower = 0; itr_tower < numTowers; itr_tower++) {
        int iEta_raw;
        int iPhi_raw;
        int towerShare;
        ss >> iEta_raw >> iPhi_raw >> towerShare;
        int splitDivisor = (subdet == 2) ? splitDivisorScint_ : splitDivisorSilic_;
        if ((towerShare > splitDivisor) || (towerShare < 1)) {
          throw edm::Exception(edm::errors::Configuration, "Configuration")
              << "HGCalTriggerTowerGeometryHelper warning: invalid tower share in the mapping file.\n"
              << "Tower share must be a positive integer and less than splitDivisor. The incorrect values found for "
                 "module ID:"
              << std::endl
              << "subdet=" << subdet << ", l=" << layer << ", u=" << moduleU << ", v=" << moduleV << std::endl;
        }
        towers.push_back(packTowerIDandShare(iEta_raw, iPhi_raw, towerShare));
      }
      modules_to_trigger_towers_[packed_modID] = towers;
    }
    moduleTowerMappingStream.close();
  }
}

unsigned HGCalTriggerTowerGeometryHelper::packLayerSubdetWaferId(int subdet, int layer, int moduleU, int moduleV) const {
  unsigned packed_modID = 0;
  packed_modID |= ((subdet & HGCalTriggerModuleDetId::kHGCalTriggerSubdetMask)
                   << HGCalTriggerModuleDetId::kHGCalTriggerSubdetOffset);
  packed_modID |= ((layer & HGCalTriggerModuleDetId::kHGCalLayerMask) << HGCalTriggerModuleDetId::kHGCalLayerOffset);
  packed_modID |=
      ((moduleU & HGCalTriggerModuleDetId::kHGCalModuleUMask) << HGCalTriggerModuleDetId::kHGCalModuleUOffset);
  packed_modID |=
      ((moduleV & HGCalTriggerModuleDetId::kHGCalModuleVMask) << HGCalTriggerModuleDetId::kHGCalModuleVOffset);
  return packed_modID;
}

unsigned HGCalTriggerTowerGeometryHelper::packTowerIDandShare(int iEta_raw, int iPhi_raw, int towerShare) const {
  unsigned packed_towerIDandShare = 0;
  unsigned iEtaAbs = std::abs(iEta_raw);
  unsigned iEtaSign = std::signbit(iEta_raw);
  unsigned iPhiAbs = std::abs(iPhi_raw);
  unsigned iPhiSign = std::signbit(iPhi_raw);
  packed_towerIDandShare |= ((iEtaAbs & l1t::HGCalTowerID::coordMask) << l1t::HGCalTowerID::coord1Shift);
  packed_towerIDandShare |= ((iEtaSign & signMask) << sign1Shift);
  packed_towerIDandShare |= ((iPhiAbs & l1t::HGCalTowerID::coordMask) << l1t::HGCalTowerID::coord2Shift);
  packed_towerIDandShare |= ((iPhiSign & signMask) << sign2Shift);
  packed_towerIDandShare |= ((towerShare & towerShareMask) << towerShareShift);
  return packed_towerIDandShare;
}

void HGCalTriggerTowerGeometryHelper::unpackTowerIDandShare(unsigned towerIDandShare,
                                                            int& iEta_raw,
                                                            int& iPhi_raw,
                                                            int& towerShare) const {
  //eta
  iEta_raw = (towerIDandShare >> l1t::HGCalTowerID::coord1Shift) & l1t::HGCalTowerID::coordMask;
  unsigned iEtaSign = (towerIDandShare >> sign1Shift) & signMask;
  iEta_raw = (iEtaSign) ? -1 * iEta_raw : iEta_raw;
  //phi
  iPhi_raw = (towerIDandShare >> l1t::HGCalTowerID::coord2Shift) & l1t::HGCalTowerID::coordMask;
  unsigned iPhiSign = (towerIDandShare >> sign2Shift) & signMask;
  iPhi_raw = (iPhiSign) ? -1 * iPhi_raw : iPhi_raw;
  //tower share
  towerShare = (towerIDandShare >> towerShareShift) & towerShareMask;
}

int HGCalTriggerTowerGeometryHelper::moveToCorrectSector(int iPhi_raw, int sector) const {
  int iPhi = (iPhi_raw + sector * rotate120Deg_ + rotate180Deg_) % int(nBinsPhi_);
  return iPhi;
}

void HGCalTriggerTowerGeometryHelper::reverseXaxis(int& iPhi) const {
  iPhi = reverseX_ - iPhi;                          //correct x -> -x in z>0
  iPhi = (int(nBinsPhi_) + iPhi) % int(nBinsPhi_);  // make all phi between 0 to nBinsPhi_-1
}

const std::vector<l1t::HGCalTowerCoord>& HGCalTriggerTowerGeometryHelper::getTowerCoordinates() const {
  return tower_coords_;
}

unsigned short HGCalTriggerTowerGeometryHelper::getTriggerTowerFromEtaPhi(const float& eta, const float& phi) const {
  auto bin_eta_l = std::lower_bound(binsEta_.begin(), binsEta_.end(), fabs(eta));
  unsigned int bin_eta = 0;
  // we add a protection for TCs in Hadron part which are outside the boundaries and possible rounding effects
  if (bin_eta_l == binsEta_.end()) {
    if (fabs(eta) < minEta_) {
      bin_eta = 0;
    } else if (fabs(eta) >= maxEta_) {
      bin_eta = nBinsEta_;
    } else {
      edm::LogError("HGCalTriggerTowerGeometryHelper")
          << " did not manage to map eta " << eta << " to any Trigger Tower\n";
    }
  } else {
    bin_eta = bin_eta_l - binsEta_.begin() - 1;
  }

  auto bin_phi_l = std::lower_bound(binsPhi_.begin(), binsPhi_.end(), phi);
  unsigned int bin_phi = 0;
  if (bin_phi_l == binsPhi_.end()) {
    if (phi < minPhi_) {
      bin_phi = nBinsPhi_;
    } else if (phi >= maxPhi_) {
      bin_phi = 0;
    } else {
      edm::LogError("HGCalTriggerTowerGeometryHelper")
          << " did not manage to map phi " << phi << " to any Trigger Tower\n";
    }
  } else {
    bin_phi = bin_phi_l - binsPhi_.begin() - 1;
  }
  int zside = eta < 0 ? -1 : 1;
  return l1t::HGCalTowerID(doNose_, zside, bin_eta, bin_phi).rawId();
}

std::unordered_map<unsigned short, float> HGCalTriggerTowerGeometryHelper::getTriggerTower(
    const l1t::HGCalTriggerCell& thecell) const {
  std::unordered_map<unsigned short, float> towerIDandShares = {};
  unsigned int trigger_cell_id = thecell.detId();
  // NOTE: if the TC is not found in the map than it is mapped via eta-phi coords.
  // this can be considered dangerous (silent failure of the map) but it actually allows to save
  // memory mapping explicitly only what is actually needed
  auto tower_id_itr = cells_to_trigger_towers_.find(trigger_cell_id);
  if (tower_id_itr != cells_to_trigger_towers_.end()) {
    towerIDandShares.insert({tower_id_itr->second, 1.0});
    return towerIDandShares;
  }
  towerIDandShares.insert({getTriggerTowerFromEtaPhi(thecell.position().eta(), thecell.position().phi()), 1.0});
  return towerIDandShares;
}

std::unordered_map<unsigned short, float> HGCalTriggerTowerGeometryHelper::getTriggerTower(
    const l1t::HGCalTriggerSums& thesum) const {
  std::unordered_map<unsigned short, float> towerIDandShares = {};
  if (!splitModuleSum_) {
    towerIDandShares.insert({getTriggerTowerFromEtaPhi(thesum.position().eta(), thesum.position().phi()), 1.0});
    return towerIDandShares;
  } else {
    HGCalTriggerModuleDetId detid(thesum.detId());
    int moduleU = detid.moduleU();
    int moduleV = detid.moduleV();
    int layer = detid.layer();
    int sector = detid.sector();
    int zside = detid.zside();
    int subdet = 0;
    int splitDivisor = splitDivisorSilic_;
    if (detid.isHScintillator()) {
      subdet = 2;
      splitDivisor = splitDivisorScint_;
    } else if (detid.isEE()) {
      subdet = 0;
      splitDivisor = splitDivisorSilic_;
    } else if (detid.isHSilicon()) {
      subdet = 1;
      splitDivisor = splitDivisorSilic_;
    } else {  //HFNose
      towerIDandShares.insert({getTriggerTowerFromEtaPhi(thesum.position().eta(), thesum.position().phi()), 1.0});
      return towerIDandShares;
    }

    unsigned packed_modID = packLayerSubdetWaferId(subdet, layer, moduleU, moduleV);
    auto module_id_itr = modules_to_trigger_towers_.find(packed_modID);
    if (module_id_itr != modules_to_trigger_towers_.end()) {
      //eta variables
      int iEta = -999;
      int iEta_raw = -999;
      int offsetEta = 2;
      //phi variables
      int iPhi = -999;
      int iPhi_raw = -999;
      int towerShare = -999;  //the share each tower gets from module sum
      for (auto towerIDandShare : module_id_itr->second) {
        unpackTowerIDandShare(towerIDandShare, iEta_raw, iPhi_raw, towerShare);
        iEta = offsetEta + iEta_raw;
        iPhi = moveToCorrectSector(iPhi_raw, sector);
        if (zside == 1) {
          reverseXaxis(iPhi);
        }
        towerIDandShares.insert(
            {l1t::HGCalTowerID(doNose_, zside, iEta, iPhi).rawId(), double(towerShare) / splitDivisor});
      }
      return towerIDandShares;
    } else {  // for modules not found in the mapping file (currently a few partial modules) use the traditional method.
      towerIDandShares.insert({getTriggerTowerFromEtaPhi(thesum.position().eta(), thesum.position().phi()), 1.0});
      return towerIDandShares;
    }
  }
}
