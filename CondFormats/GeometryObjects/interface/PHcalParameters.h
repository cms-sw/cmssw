#ifndef CondFormats_GeometryObjects_PHcalParameters_h
#define CondFormats_GeometryObjects_PHcalParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

class PHcalParameters
{
 public:
  
  PHcalParameters( void ) { }
  ~PHcalParameters( void ) { }

  struct LayerItem
  {
    unsigned int layer;
    std::vector<int> layerGroup;

    COND_SERIALIZABLE;
  };

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
  std::vector<int> noff;
  std::vector<int> etaMin;
  std::vector<int> etaMax;
  std::vector<int> HBShift;
  std::vector<int> HEShift;
  std::vector<int> HFShift;

  std::vector<int> etagroup;
  std::vector<int> phigroup;
  std::vector<LayerItem> layerGroupEta;

  int topologyMode;

  COND_SERIALIZABLE;
};

#endif
