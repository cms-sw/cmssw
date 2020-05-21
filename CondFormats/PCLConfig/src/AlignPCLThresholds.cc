#include "CondFormats/PCLConfig/interface/AlignPCLThresholds.h"
#include "CondFormats/PCLConfig/interface/AlignPCLThreshold.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>  // std::setw

//****************************************************************************//
void AlignPCLThresholds::setAlignPCLThreshold(const std::string &AlignableId, const AlignPCLThreshold &Threshold) {
  m_thresholds[AlignableId] = Threshold;
}

//****************************************************************************//
void AlignPCLThresholds::setAlignPCLThresholds(const int &Nrecords, const threshold_map &AlignPCLThresholds) {
  m_nrecords = Nrecords;
  m_thresholds = AlignPCLThresholds;
}

//****************************************************************************//
void AlignPCLThresholds::setNRecords(const int &Nrecords) { m_nrecords = Nrecords; }

//****************************************************************************//
AlignPCLThreshold AlignPCLThresholds::getAlignPCLThreshold(const std::string &AlignableId) const {
  threshold_map::const_iterator it = m_thresholds.find(AlignableId);

  if (it != m_thresholds.end()) {
    return it->second;
  } else {
    throw cms::Exception("AlignPCLThresholds") << "No Thresholds defined for Alignable id " << AlignableId << "\n";
  }
}

//****************************************************************************//
AlignPCLThreshold &AlignPCLThresholds::getAlignPCLThreshold(const std::string &AlignableId) {
  return m_thresholds[AlignableId];
}

float AlignPCLThresholds::getSigCut(const std::string &AlignableId, const coordType &type) const {
  AlignPCLThreshold a = getAlignPCLThreshold(AlignableId);
  switch (type) {
    case X:
      return a.getSigXcut();
    case Y:
      return a.getSigYcut();
    case Z:
      return a.getSigZcut();
    case theta_X:
      return a.getSigThetaXcut();
    case theta_Y:
      return a.getSigThetaYcut();
    case theta_Z:
      return a.getSigThetaZcut();
    default:
      throw cms::Exception("AlignPCLThresholds")
          << "Requested significance threshold for undefined coordinate" << type << "\n";
  }
}

//****************************************************************************//
// overloaded method
//****************************************************************************//
std::array<float, 6> AlignPCLThresholds::getSigCut(const std::string &AlignableId) const {
  AlignPCLThreshold a = getAlignPCLThreshold(AlignableId);
  return {
      {a.getSigXcut(), a.getSigYcut(), a.getSigZcut(), a.getSigThetaXcut(), a.getSigThetaYcut(), a.getSigThetaZcut()}};
}

float AlignPCLThresholds::getCut(const std::string &AlignableId, const coordType &type) const {
  AlignPCLThreshold a = getAlignPCLThreshold(AlignableId);
  switch (type) {
    case X:
      return a.getXcut();
    case Y:
      return a.getYcut();
    case Z:
      return a.getZcut();
    case theta_X:
      return a.getThetaXcut();
    case theta_Y:
      return a.getThetaYcut();
    case theta_Z:
      return a.getThetaZcut();
    default:
      throw cms::Exception("AlignPCLThresholds")
          << "Requested significance threshold for undefined coordinate" << type << "\n";
  }
}

//****************************************************************************//
// overloaded method
//****************************************************************************//
std::array<float, 6> AlignPCLThresholds::getCut(const std::string &AlignableId) const {
  AlignPCLThreshold a = getAlignPCLThreshold(AlignableId);
  return {{a.getXcut(), a.getYcut(), a.getZcut(), a.getThetaXcut(), a.getThetaYcut(), a.getThetaZcut()}};
}

//****************************************************************************//
float AlignPCLThresholds::getMaxMoveCut(const std::string &AlignableId, const coordType &type) const {
  AlignPCLThreshold a = getAlignPCLThreshold(AlignableId);
  switch (type) {
    case X:
      return a.getMaxMoveXcut();
    case Y:
      return a.getMaxMoveYcut();
    case Z:
      return a.getMaxMoveZcut();
    case theta_X:
      return a.getMaxMoveThetaXcut();
    case theta_Y:
      return a.getMaxMoveThetaYcut();
    case theta_Z:
      return a.getMaxMoveThetaZcut();
    default:
      throw cms::Exception("AlignPCLThresholds")
          << "Requested significance threshold for undefined coordinate" << type << "\n";
  }
}

//****************************************************************************//
// overloaded method
//****************************************************************************//
std::array<float, 6> AlignPCLThresholds::getMaxMoveCut(const std::string &AlignableId) const {
  AlignPCLThreshold a = getAlignPCLThreshold(AlignableId);
  return {{a.getMaxMoveXcut(),
           a.getMaxMoveYcut(),
           a.getMaxMoveZcut(),
           a.getMaxMoveThetaXcut(),
           a.getMaxMoveThetaYcut(),
           a.getMaxMoveThetaZcut()}};
}

//****************************************************************************//
float AlignPCLThresholds::getMaxErrorCut(const std::string &AlignableId, const coordType &type) const {
  AlignPCLThreshold a = getAlignPCLThreshold(AlignableId);
  switch (type) {
    case X:
      return a.getErrorXcut();
    case Y:
      return a.getErrorYcut();
    case Z:
      return a.getErrorZcut();
    case theta_X:
      return a.getErrorThetaXcut();
    case theta_Y:
      return a.getErrorThetaYcut();
    case theta_Z:
      return a.getErrorThetaZcut();
    default:
      throw cms::Exception("AlignPCLThresholds")
          << "Requested significance threshold for undefined coordinate" << type << "\n";
  }
}

//****************************************************************************//
// overloaded method
//****************************************************************************//
std::array<float, 6> AlignPCLThresholds::getMaxErrorCut(const std::string &AlignableId) const {
  AlignPCLThreshold a = getAlignPCLThreshold(AlignableId);
  return {{a.getErrorXcut(),
           a.getErrorYcut(),
           a.getErrorZcut(),
           a.getErrorThetaXcut(),
           a.getErrorThetaYcut(),
           a.getErrorThetaZcut()}};
}

//****************************************************************************//
std::array<float, 4> AlignPCLThresholds::getExtraDOFCutsForAlignable(const std::string &AlignableId,
                                                                     const unsigned int i) const {
  AlignPCLThreshold a = getAlignPCLThreshold(AlignableId);
  return a.getExtraDOFCuts(i);
}

//****************************************************************************//
std::string AlignPCLThresholds::getExtraDOFLabelForAlignable(const std::string &AlignableId,
                                                             const unsigned int i) const {
  AlignPCLThreshold a = getAlignPCLThreshold(AlignableId);
  return a.getExtraDOFLabel(i);
}

//****************************************************************************//
void AlignPCLThresholds::printAll() const {
  edm::LogVerbatim("AlignPCLThresholds") << "AlignPCLThresholds::printAll()";
  edm::LogVerbatim("AlignPCLThresholds") << " ========================================================================="
                                            "==========================================";
  edm::LogVerbatim("AlignPCLThresholds") << "N records cut: " << this->getNrecords();
  for (const auto &m_threshold : m_thresholds) {
    edm::LogVerbatim("AlignPCLThresholds") << " ======================================================================="
                                              "============================================";
    edm::LogVerbatim("AlignPCLThresholds")
        << "key : " << m_threshold.first << " \n"
        << "- Xcut             : " << std::setw(4) << (m_threshold.second).getXcut() << std::setw(5) << "   um"
        << "| sigXcut          : " << std::setw(4) << (m_threshold.second).getSigXcut() << std::setw(1) << " "
        << "| maxMoveXcut      : " << std::setw(4) << (m_threshold.second).getMaxMoveXcut() << std::setw(5) << "   um"
        << "| ErrorXcut        : " << std::setw(4) << (m_threshold.second).getErrorXcut() << std::setw(5) << "   um\n"

        << "- thetaXcut        : " << std::setw(4) << (m_threshold.second).getThetaXcut() << std::setw(5) << " urad"
        << "| sigThetaXcut     : " << std::setw(4) << (m_threshold.second).getSigThetaXcut() << std::setw(1) << " "
        << "| maxMoveThetaXcut : " << std::setw(4) << (m_threshold.second).getMaxMoveThetaXcut() << std::setw(5)
        << " urad"
        << "| ErrorThetaXcut   : " << std::setw(4) << (m_threshold.second).getErrorThetaXcut() << std::setw(5)
        << " urad\n"

        << "- Ycut             : " << std::setw(4) << (m_threshold.second).getYcut() << std::setw(5) << "   um"
        << "| sigYcut          : " << std::setw(4) << (m_threshold.second).getSigXcut() << std::setw(1) << " "
        << "| maxMoveYcut      : " << std::setw(4) << (m_threshold.second).getMaxMoveYcut() << std::setw(5) << "   um"
        << "| ErrorYcut        : " << std::setw(4) << (m_threshold.second).getErrorYcut() << std::setw(5) << "   um\n"

        << "- thetaYcut        : " << std::setw(4) << (m_threshold.second).getThetaYcut() << std::setw(5) << " urad"
        << "| sigThetaYcut     : " << std::setw(4) << (m_threshold.second).getSigThetaYcut() << std::setw(1) << " "
        << "| maxMoveThetaYcut : " << std::setw(4) << (m_threshold.second).getMaxMoveThetaYcut() << std::setw(5)
        << " urad"
        << "| ErrorThetaYcut   : " << std::setw(4) << (m_threshold.second).getErrorThetaYcut() << std::setw(5)
        << " urad\n"

        << "- Zcut             : " << std::setw(4) << (m_threshold.second).getZcut() << std::setw(5) << "   um"
        << "| sigZcut          : " << std::setw(4) << (m_threshold.second).getSigZcut() << std::setw(1) << " "
        << "| maxMoveZcut      : " << std::setw(4) << (m_threshold.second).getMaxMoveZcut() << std::setw(5) << "   um"
        << "| ErrorZcut        : " << std::setw(4) << (m_threshold.second).getErrorZcut() << std::setw(5) << "   um\n"

        << "- thetaZcut        : " << std::setw(4) << (m_threshold.second).getThetaZcut() << std::setw(5) << " urad"
        << "| sigThetaZcut     : " << std::setw(4) << (m_threshold.second).getSigThetaZcut() << std::setw(1) << " "
        << "| maxMoveThetaZcut : " << std::setw(4) << (m_threshold.second).getMaxMoveThetaZcut() << std::setw(5)
        << " urad"
        << "| ErrorThetaZcut   : " << std::setw(4) << (m_threshold.second).getErrorThetaZcut() << std::setw(5)
        << " urad";

    if ((m_threshold.second).hasExtraDOF()) {
      for (unsigned int j = 0; j < (m_threshold.second).extraDOFSize(); j++) {
        std::array<float, 4> extraDOFCuts = getExtraDOFCutsForAlignable(m_threshold.first, j);

        edm::LogVerbatim("AlignPCLThresholds")
            << "Extra DOF " << j << " with label: " << getExtraDOFLabelForAlignable(m_threshold.first, j);
        edm::LogVerbatim("AlignPCLThresholds")
            << "- cut              : " << std::setw(4) << extraDOFCuts.at(0) << std::setw(5) << "    "
            << "| sigCut           : " << std::setw(4) << extraDOFCuts.at(1) << std::setw(1) << " "
            << "| maxMoveCut       : " << std::setw(4) << extraDOFCuts.at(2) << std::setw(5) << "    "
            << "| maxErrorCut      : " << std::setw(4) << extraDOFCuts.at(3) << std::setw(5) << "    ";
      }
    }
  }
}

//****************************************************************************//
std::vector<std::string> AlignPCLThresholds::getAlignableList() const {
  std::vector<std::string> alignables_;
  alignables_.reserve(m_thresholds.size());

  for (const auto &m_threshold : m_thresholds) {
    alignables_.push_back(m_threshold.first);
  }
  return alignables_;
}
