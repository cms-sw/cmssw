#ifndef CondFormats_GeometryObjects_HcalParameters_h
#define CondFormats_GeometryObjects_HcalParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

class HcalParameters {
public:
  HcalParameters(void) = default;
  ~HcalParameters(void) = default;

  struct LayerItem {
    unsigned int layer;
    std::vector<int> layerGroup;
    COND_SERIALIZABLE;
  };

  std::vector<double> rHB;
  std::vector<double> drHB;
  std::vector<double> zHE;
  std::vector<double> dzHE;
  std::vector<double> zHO;

  std::vector<double> rhoxHB;
  std::vector<double> zxHB;
  std::vector<double> dyHB;
  std::vector<double> dxHB;
  std::vector<double> rhoxHE;
  std::vector<double> zxHE;
  std::vector<double> dyHE;
  std::vector<double> dx1HE;
  std::vector<double> dx2HE;
  std::vector<double> rHO;

  std::vector<double> phioff;
  std::vector<double> etaTable;
  std::vector<double> rTable;
  std::vector<double> phibin;
  std::vector<double> phitable;
  std::vector<double> etaRange;
  std::vector<double> gparHF;
  std::vector<double> Layer0Wt;
  std::vector<double> HBGains;
  std::vector<double> HEGains;
  std::vector<double> HFGains;
  std::vector<double> etaTableHF;
  double dzVcal;

  std::vector<int> maxDepth;
  std::vector<int> modHB;
  std::vector<int> modHE;
  std::vector<int> layHB;
  std::vector<int> layHE;

  std::vector<int> etaMin;
  std::vector<int> etaMax;
  std::vector<int> noff;
  std::vector<int> HBShift;
  std::vector<int> HEShift;
  std::vector<int> HFShift;

  std::vector<int> etagroup;
  std::vector<int> phigroup;
  std::vector<LayerItem> layerGroupEtaSim, layerGroupEtaRec;
  int topologyMode;

  uint32_t etaMaxHBHE() const { return static_cast<uint32_t>(etagroup.size()); }
  COND_SERIALIZABLE;
};

#endif
