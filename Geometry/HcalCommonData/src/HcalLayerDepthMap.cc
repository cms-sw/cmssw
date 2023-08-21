#include "Geometry/HcalCommonData/interface/HcalLayerDepthMap.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>
#include <sstream>

//#define EDM_ML_DEBUG

HcalLayerDepthMap::HcalLayerDepthMap() {
  subdet_ = 0;
  ietaMin_ = ietaMax_ = 0;
  depthMin_ = 99;
  depthMax_ = -1;
  dep29C_ = 2;
  wtl0C_ = 1.;
}

HcalLayerDepthMap::~HcalLayerDepthMap() {}

void HcalLayerDepthMap::initialize(const int subdet,
                                   const int ietaMax,
                                   const int dep16C,
                                   const int dep29C,
                                   const double wtl0C,
                                   std::vector<int> const& iphi,
                                   std::vector<int> const& ieta,
                                   std::vector<int> const& layer,
                                   std::vector<int> const& depth) {
  subdet_ = subdet;
  ietaMin_ = ietaMax_ = ietaMax;
  dep16C_ = dep16C;
  dep29C_ = dep29C;
  wtl0C_ = wtl0C;
  iphi_.insert(iphi_.end(), iphi.begin(), iphi.end());
  layer2Depth_.clear();
  depth2LayerF_.clear();
  depth2LayerB_.clear();
  depthMin_ = 99;
  depthMax_ = -1;
  for (unsigned int k = 0; k < ieta.size(); ++k) {
    if (ieta[k] < ietaMin_)
      ietaMin_ = ieta[k];
    if (depth[k] < depthMin_)
      depthMin_ = depth[k];
    if (depth[k] > depthMax_)
      depthMax_ = depth[k];
  }
  //Assume ieta, layer, depth are in increasing order of ieta and depth
  for (unsigned int k1 = 0; k1 < ieta.size(); ++k1) {
    int ietaMin = ieta[k1];
    int ietaMax = ietaMax_;
    int layMin = layer[k1];
    int layMax = (k1 + 1 < ieta.size()) ? (layer[k1 + 1] - 1) : maxLayers_;
    for (unsigned int k2 = k1 + 1; k2 < ieta.size(); ++k2) {
      if (ieta[k2] > ieta[k1]) {
        ietaMax = ieta[k2] - 1;
        if (k2 == k1 + 1)
          layMax = maxLayers_;
        break;
      }
    }
    for (int eta = ietaMin; eta <= ietaMax; ++eta) {
      depth2LayerF_[std::pair<int, int>(eta, depth[k1])] = layMin;
      depth2LayerB_[std::pair<int, int>(eta, depth[k1])] = layMax;
      for (int lay = layMin; lay <= layMax; ++lay)
        layer2Depth_[std::pair<int, int>(eta, lay)] = depth[k1];
    }
  }
  for (int eta = ietaMin_; eta <= ietaMax_; ++eta) {
    int dmin(99), dmax(-1);
    for (auto& itr : layer2Depth_) {
      if ((itr.first).first == eta) {
        if ((itr.second) < dmin)
          dmin = (itr.second);
        if ((itr.second) > dmax)
          dmax = (itr.second);
      }
    }
    if (subdet == 2) {
      if (eta == ietaMin_)
        dmin = dep16C_;
      else if (eta == ietaMax_)
        dmax = dep29C_;
    }
    depthsEta_[eta] = std::pair<int, int>(dmin, dmax);
  }
#ifdef EDM_ML_DEBUG
  std::ostringstream st1;
  st1 << "HcalLayerDepthMap: Subdet " << subdet_ << " iEta " << ietaMin_ << ":" << ietaMax_ << " depth " << depthMin_
      << ":" << depthMax_ << "\nMaximum Depth for last HE towers " << dep29C_ << " Layer 0 Weight " << wtl0C_
      << " iPhi";
  for (unsigned int k = 0; k < iphi_.size(); ++k)
    st1 << ":" << iphi_[k];
  edm::LogVerbatim("HCalGeom") << st1.str();
  edm::LogVerbatim("HCalGeom") << "Layer2Depth_ with " << layer2Depth_.size() << " elements" << std::endl;
  for (std::map<std::pair<int, int>, int>::iterator itr = layer2Depth_.begin(); itr != layer2Depth_.end(); ++itr)
    edm::LogVerbatim("HCalGeom") << "iEta " << (itr->first).first << " Layer " << (itr->first).second << " Depth "
                                 << itr->second;
  edm::LogVerbatim("HCalGeom") << "Depth2LayerFront with " << depth2LayerF_.size() << " elemsts";
  for (std::map<std::pair<int, int>, int>::iterator itr = depth2LayerF_.begin(); itr != depth2LayerF_.end(); ++itr)
    edm::LogVerbatim("HCalGeom") << "iEta " << (itr->first).first << " Depth " << (itr->first).second << " Layer "
                                 << itr->second;
  edm::LogVerbatim("HCalGeom") << "Depth2LayerBack with " << depth2LayerB_.size() << " elemets";
  for (std::map<std::pair<int, int>, int>::iterator itr = depth2LayerB_.begin(); itr != depth2LayerB_.end(); ++itr)
    edm::LogVerbatim("HCalGeom") << "iEta " << (itr->first).first << " Depth " << (itr->first).second << " Layer "
                                 << itr->second;
  edm::LogVerbatim("HCalGeom") << "DepthsEta_ with " << depthsEta_.size() << " elements";
  for (std::map<int, std::pair<int, int> >::iterator itr = depthsEta_.begin(); itr != depthsEta_.end(); ++itr)
    edm::LogVerbatim("HCalGeom") << "iEta " << itr->first << " Depths " << (itr->second).first << ":"
                                 << (itr->second).second;
#endif
}

int HcalLayerDepthMap::getDepth(
    const int subdet, const int ieta, const int iphi, const int zside, const int layer) const {
  int depth(-1);
  if (isValid(subdet, iphi, zside)) {
    std::map<std::pair<int, int>, int>::const_iterator itr = layer2Depth_.find(std::pair<int, int>(ieta, layer));
    if (itr != layer2Depth_.end())
      depth = itr->second;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Debug Info -- getDepth::Input " << subdet << ":" << ieta << ":" << iphi << ":"
                               << zside << ":" << layer << " Output " << depth;
#endif
  return depth;
}

int HcalLayerDepthMap::getDepth16(const int subdet, const int iphi, const int zside) const {
  int depth(-1);
  if (isValid(subdet, iphi, zside))
    depth = dep16C_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Debug info -- getDepth16::Input " << subdet << ":" << iphi << ":" << zside
                               << " Output " << depth;
#endif
  return depth;
}

int HcalLayerDepthMap::getDepthMin(const int subdet, const int iphi, const int zside) const {
  int depth = (isValid(subdet, iphi, zside)) ? depthMin_ : -1;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Debug info -- getDepthMin::Input " << subdet << ":" << iphi << ":" << zside
                               << " Output " << depth;
#endif
  return depth;
}

int HcalLayerDepthMap::getDepthMax(const int subdet, const int iphi, const int zside) const {
  int depth = (isValid(subdet, iphi, zside)) ? depthMax_ : -1;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Debug info -- getDepthMax::Input " << subdet << ":" << iphi << ":" << zside
                               << " Output " << depth;
#endif
  return depth;
}

int HcalLayerDepthMap::getDepthMax(const int subdet, const int ieta, const int iphi, const int zside) const {
  int depth = (isValid(subdet, iphi, zside)) ? getDepth(subdet, ieta, iphi, zside, maxLayers_) : -1;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Debug info -- getDepthMax::Input " << subdet << ":" << iphi << ":" << zside
                               << " Output " << depth;
#endif
  return depth;
}

std::pair<int, int> HcalLayerDepthMap::getDepths(const int eta) const {
  std::map<int, std::pair<int, int> >::const_iterator itr = depthsEta_.find(eta);
  if (itr == depthsEta_.end())
    return std::pair<int, int>(-1, -1);
  else
    return itr->second;
}

int HcalLayerDepthMap::getLayerFront(
    const int subdet, const int ieta, const int iphi, const int zside, const int depth) const {
  int layer(-1);
  if (isValid(subdet, iphi, zside)) {
    std::map<std::pair<int, int>, int>::const_iterator itr = depth2LayerF_.find(std::pair<int, int>(ieta, depth));
    if (itr != depth2LayerF_.end())
      layer = itr->second;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Debug info -- getLayerFront::Input " << subdet << ":" << ieta << ":" << iphi << ":"
                               << zside << ":" << depth << " Output " << layer;
#endif
  return layer;
}

int HcalLayerDepthMap::getLayerBack(
    const int subdet, const int ieta, const int iphi, const int zside, const int depth) const {
  int layer(-1);
  if (isValid(subdet, iphi, zside)) {
    std::map<std::pair<int, int>, int>::const_iterator itr = depth2LayerB_.find(std::pair<int, int>(ieta, depth));
    if (itr != depth2LayerB_.end())
      layer = itr->second;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Debug info -- getLayerBack::Input " << subdet << ":" << ieta << ":" << iphi << ":"
                               << zside << ":" << depth << " Output " << layer;
#endif
  return layer;
}

void HcalLayerDepthMap::getLayerDepth(
    const int subdet, const int eta, const int phi, const int zside, std::map<int, int>& layers) const {
  layers.clear();
  if (isValid(subdet, phi, zside)) {
    for (const auto& itr : layer2Depth_) {
      if ((itr.first).first == eta) {
        layers[((itr.first).second) + 1] = (itr.second);
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Debug info -- getLayerDepth::Input " << subdet << ":" << eta << ":" << phi << ":"
                               << zside << " Output " << layers.size() << " entries";
  std::ostringstream st1;
  for (std::map<int, int>::iterator itr = layers.begin(); itr != layers.end(); ++itr)
    st1 << " [" << itr->first << "] " << itr->second;
  edm::LogVerbatim("HCalGeom") << st1.str();
#endif
}

void HcalLayerDepthMap::getLayerDepth(const int eta, std::map<int, int>& layers) const {
  layers.clear();
  if (subdet_ > 0) {
    for (const auto& itr : layer2Depth_) {
      if ((itr.first).first == eta) {
        layers[((itr.first).second) + 1] = (itr.second);
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Debug info -- getLayerDepth::Input " << eta << " Output " << layers.size()
                               << " entries";
  std::ostringstream st1;
  for (std::map<int, int>::iterator itr = layers.begin(); itr != layers.end(); ++itr)
    st1 << " [" << itr->first << "] " << itr->second;
  edm::LogVerbatim("HCalGeom") << st1.str();
#endif
}

int HcalLayerDepthMap::getMaxDepthLastHE(const int subdet, const int iphi, const int zside) const {
  int depth = isValid(subdet, iphi, zside) ? dep29C_ : -1;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Debug info -- getMaxDepthLastHE::Input " << subdet << ":" << iphi << ":" << zside
                               << " Output " << depth;
#endif
  return depth;
}

double HcalLayerDepthMap::getLayer0Wt(const int subdet, const int iphi, const int zside) const {
  double wt = isValid(subdet, iphi, zside) ? wtl0C_ : -1.0;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalLayerDepthMap -- getLayer0Wt::Input " << subdet << ":" << iphi << ":" << zside
                               << " Output " << wt;
#endif
  return wt;
}

bool HcalLayerDepthMap::isValid(const int subdet, const int iphi, const int zside) const {
  bool flag(false);
  int kphi = (zside > 0) ? iphi : -iphi;
  if (subdet == subdet_)
    flag = (std::find(iphi_.begin(), iphi_.end(), kphi) != iphi_.end());
  return flag;
}

int HcalLayerDepthMap::validDet(std::vector<int>& phi) const {
  phi.insert(phi.end(), iphi_.begin(), iphi_.end());
  return subdet_;
}
