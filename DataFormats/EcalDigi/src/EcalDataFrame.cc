#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"

int EcalDataFrame::lastUnsaturatedSample() const {
  int cnt = 0;
  for (size_t i = 3; i < m_data.size(); ++i) {
    cnt = 0;
    for (size_t j = i; j < (i + 5) && j < m_data.size(); ++j) {
      if (((EcalMGPASample)m_data[j]).gainId() == EcalMgpaBitwiseGain0)
        ++cnt;
    }
    if (cnt == 5)
      return i - 1;  // the last unsaturated sample
  }
  return -1;  // no saturation found
}

bool EcalDataFrame::hasSwitchToGain6() const {
  for (unsigned int u = 0; u < m_data.size(); u++) {
    if ((static_cast<EcalMGPASample>(m_data[u])).gainId() == EcalMgpaBitwiseGain6)
      return true;
  }
  return false;
}

bool EcalDataFrame::hasSwitchToGain1() const {
  for (unsigned int u = 0; u < m_data.size(); u++) {
    if ((static_cast<EcalMGPASample>(m_data[u])).gainId() == EcalMgpaBitwiseGain1)
      return true;
  }
  return false;
}
