#include "CondFormats/MuonSystemAging/interface/MuonSystemAging.h"
MuonSystemAging::MuonSystemAging(){
  m_RPCchambers.reserve(600000);
  m_DTchambers.reserve(600000);
  m_CSCineff = 0.0;
  m_GE11Pluschambers.reserve(600000);
  m_GE11Minuschambers.reserve(600000);
  m_GE21Pluschambers.reserve(600000);
  m_GE21Minuschambers.reserve(600000);
  m_ME0Pluschambers.reserve(600000);
  m_ME0Minuschambers.reserve(600000);

}
