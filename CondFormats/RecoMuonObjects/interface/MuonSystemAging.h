#ifndef MuonSystemAging_H
#define MuonSystemAging_H

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <regex>
#include <map>

enum CSCInefficiencyType { EFF_CHAMBER = 0, EFF_STRIPS = 1, EFF_WIRES = 2 };

class MuonSystemAging {
public:
  MuonSystemAging(){};
  ~MuonSystemAging(){};

  std::map<unsigned int, float> m_RPCChambEffs;
  std::map<unsigned int, float> m_DTChambEffs;
  std::map<unsigned int, std::pair<unsigned int, float> > m_CSCChambEffs;

  std::map<unsigned int, float> m_GEMChambEffs;
  std::map<unsigned int, float> m_ME0ChambEffs;

  COND_SERIALIZABLE;
};

#endif
