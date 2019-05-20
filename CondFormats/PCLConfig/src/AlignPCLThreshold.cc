#include "CondFormats/PCLConfig/interface/AlignPCLThreshold.h"
#include "FWCore/Utilities/interface/Exception.h"

AlignPCLThreshold::AlignPCLThreshold(coordThresholds X,
                                     coordThresholds tX,
                                     coordThresholds Y,
                                     coordThresholds tY,
                                     coordThresholds Z,
                                     coordThresholds tZ,
                                     std::vector<coordThresholds> extraDOF) {
  m_xCoord = X;
  m_yCoord = Y;
  m_zCoord = Z;
  m_thetaXCoord = tX;
  m_thetaYCoord = tY;
  m_thetaZCoord = tZ;
  m_extraDOF = extraDOF;
};

//****************************************************************************//
std::array<float, 4> AlignPCLThreshold::getExtraDOFCuts(const unsigned int i) const {
  if (i < m_extraDOF.size()) {
    return {{m_extraDOF[i].m_Cut, m_extraDOF[i].m_sigCut, m_extraDOF[i].m_errorCut, m_extraDOF[i].m_maxMoveCut}};
  } else {
    throw cms::Exception("AlignPCLThreshold") << "No extra DOF thresholds defined for index" << i << "\n";
  }
}

//****************************************************************************//
std::string AlignPCLThreshold::getExtraDOFLabel(const unsigned int i) const {
  if (i < m_extraDOF.size()) {
    return m_extraDOF[i].m_label;
  } else {
    throw cms::Exception("AlignPCLThreshold") << "No extra DOF label defined for index" << i << "\n";
  }
}
