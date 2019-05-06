#ifndef TPedResult_h
#define TPedResult_h

/**
 * \file TPedResult.h
 * \brief Transient container
 * right DAC values for each crystal and each gain
 * $Date:
 * $Revision:
 * \author P. Govoni (pietro.govoni@cernNOSPAM.ch)
 */

#include <vector>

class TPedResult {
public:
  TPedResult() { reset(); }

  void reset() {
    for (int gainId = 1; gainId < 4; ++gainId)
      for (int crystal = 0; crystal < 1700; ++crystal)
        m_DACvalue[gainId - 1][crystal] = 0;
  }

  int m_DACvalue[3][1700];
};

#endif
