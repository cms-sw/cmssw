#include "CondFormats/PCLConfig/interface/AlignPCLThresholdsHG.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <iomanip>  // std::setw

//****************************************************************************//
namespace AlignPCLThresholdsHGImpl {
  template <typename T>
  const T &getParam(const std::vector<T> &params, size_t index) {
    if (index >= params.size())
      throw std::out_of_range("Parameter with index " + std::to_string(index) + " is out of range.");
    return params[index];
  }

  template <typename T>
  void setParam(std::vector<T> &params, size_t index, const T &value) {
    if (index >= params.size())
      throw std::out_of_range("Parameter with index " + std::to_string(index) + " is out of range.");
    params[index] = value;
  }

}  //namespace AlignPCLThresholdsHGImpl

//****************************************************************************//
void AlignPCLThresholdsHG::setFloatMap(const std::unordered_map<std::string, std::vector<float>> &floatMap) {
  floatMap_ = floatMap;
}

//****************************************************************************//
const std::vector<float> &AlignPCLThresholdsHG::getFloatVec(const std::string &AlignableId) const {
  const auto &it = floatMap_.find(AlignableId);

  if (it != floatMap_.end()) {
    return it->second;
  } else {
    throw cms::Exception("AlignPCLThresholdsHG") << "No float vector defined for Alignable id " << AlignableId << "\n";
  }
}

//****************************************************************************//
void AlignPCLThresholdsHG::setFractionCut(const std::string &AlignableId, const coordType &type, const float &cut) {
  // Set entry in map if not yet available
  const auto &it = floatMap_.find(AlignableId);
  if (it == floatMap_.end())
    floatMap_[AlignableId] = std::vector<float>(FSIZE, -1.);

  switch (type) {
    case X:
      return AlignPCLThresholdsHGImpl::setParam(floatMap_[AlignableId], FRACTION_CUT_X, cut);
    case Y:
      return AlignPCLThresholdsHGImpl::setParam(floatMap_[AlignableId], FRACTION_CUT_Y, cut);
    case Z:
      return AlignPCLThresholdsHGImpl::setParam(floatMap_[AlignableId], FRACTION_CUT_Z, cut);
    case theta_X:
      return AlignPCLThresholdsHGImpl::setParam(floatMap_[AlignableId], FRACTION_CUT_TX, cut);
    case theta_Y:
      return AlignPCLThresholdsHGImpl::setParam(floatMap_[AlignableId], FRACTION_CUT_TY, cut);
    case theta_Z:
      return AlignPCLThresholdsHGImpl::setParam(floatMap_[AlignableId], FRACTION_CUT_TZ, cut);
    default:
      throw cms::Exception("AlignPCLThresholdsHG")
          << "Requested setting fraction threshold for undefined coordinate" << type << "\n";
  }
}

//****************************************************************************//
std::array<float, 6> AlignPCLThresholdsHG::getFractionCut(const std::string &AlignableId) const {
  const std::vector<float> vec = getFloatVec(AlignableId);
  return {{AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_X),
           AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_Y),
           AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_Z),
           AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_TX),
           AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_TY),
           AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_TZ)}};
}

//****************************************************************************//
float AlignPCLThresholdsHG::getFractionCut(const std::string &AlignableId, const coordType &type) const {
  const std::vector<float> vec = getFloatVec(AlignableId);
  switch (type) {
    case X:
      return AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_X);
    case Y:
      return AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_Y);
    case Z:
      return AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_Z);
    case theta_X:
      return AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_TX);
    case theta_Y:
      return AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_TY);
    case theta_Z:
      return AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_TZ);
    default:
      throw cms::Exception("AlignPCLThresholdsHG")
          << "Requested fraction threshold for undefined coordinate" << type << "\n";
  }
}

//****************************************************************************//
const bool AlignPCLThresholdsHG::hasFloatMap(const std::string &AlignableId) const {
  const auto &it = floatMap_.find(AlignableId);

  if (it != floatMap_.end()) {
    return true;
  } else {
    return false;
  }
}

//****************************************************************************//
const int AlignPCLThresholdsHG::payloadVersion() const {
  switch (FSIZE + ISIZE + SSIZE) {
    case 6:
      return 1;
    default:
      throw cms::Exception("AlignPCLThresholdsHG")
          << "Payload version with parameter size equal to " << FSIZE + ISIZE + SSIZE << " is not defined.\n";
  }
}

//****************************************************************************//
void AlignPCLThresholdsHG::printAll() const {
  edm::LogVerbatim out("AlignPCLThresholdsHG");

  out << "AlignPCLThresholdsHG::printAll()\n";
  out << "============================================================================================================="
         "======\n";
  out << "N records cut: " << this->getNrecords() << "\n";
  for (auto it = m_thresholds.begin(); it != m_thresholds.end(); ++it) {
    out << "==========================================================================================================="
           "========\n";

    std::stringstream ss;

    ss << "key : " << it->first << " \n"
       << "- Xcut             : " << std::setw(4) << (it->second).getXcut() << std::setw(5) << "   um"
       << "| sigXcut          : " << std::setw(4) << (it->second).getSigXcut() << std::setw(1) << " "
       << "| maxMoveXcut      : " << std::setw(4) << (it->second).getMaxMoveXcut() << std::setw(5) << "   um"
       << "| ErrorXcut        : " << std::setw(4) << (it->second).getErrorXcut() << std::setw(5) << "   um";

    if (floatMap_.find(it->first) != floatMap_.end()) {
      ss << "| X_fractionCut      : " << std::setw(4) << getFractionCut(it->first, X) << std::setw(5) << "\n";
    } else {
      ss << "\n";
    }

    ss << "- thetaXcut        : " << std::setw(4) << (it->second).getThetaXcut() << std::setw(5) << " urad"
       << "| sigThetaXcut     : " << std::setw(4) << (it->second).getSigThetaXcut() << std::setw(1) << " "
       << "| maxMoveThetaXcut : " << std::setw(4) << (it->second).getMaxMoveThetaXcut() << std::setw(5) << " urad"
       << "| ErrorThetaXcut   : " << std::setw(4) << (it->second).getErrorThetaXcut() << std::setw(5) << " urad";

    if (floatMap_.find(it->first) != floatMap_.end()) {
      ss << "| thetaX_fractionCut : " << std::setw(4) << getFractionCut(it->first, theta_X) << std::setw(5) << "\n";
    } else {
      ss << "\n";
    }

    ss << "- Ycut             : " << std::setw(4) << (it->second).getYcut() << std::setw(5) << "   um"
       << "| sigYcut          : " << std::setw(4) << (it->second).getSigXcut() << std::setw(1) << " "
       << "| maxMoveYcut      : " << std::setw(4) << (it->second).getMaxMoveYcut() << std::setw(5) << "   um"
       << "| ErrorYcut        : " << std::setw(4) << (it->second).getErrorYcut() << std::setw(5) << "   um";

    if (floatMap_.find(it->first) != floatMap_.end()) {
      ss << "| Y_fractionCut      : " << std::setw(4) << getFractionCut(it->first, Y) << std::setw(5) << "\n";
    } else {
      ss << "\n";
    }

    ss << "- thetaYcut        : " << std::setw(4) << (it->second).getThetaYcut() << std::setw(5) << " urad"
       << "| sigThetaYcut     : " << std::setw(4) << (it->second).getSigThetaYcut() << std::setw(1) << " "
       << "| maxMoveThetaYcut : " << std::setw(4) << (it->second).getMaxMoveThetaYcut() << std::setw(5) << " urad"
       << "| ErrorThetaYcut   : " << std::setw(4) << (it->second).getErrorThetaYcut() << std::setw(5) << " urad";

    if (floatMap_.find(it->first) != floatMap_.end()) {
      ss << "| thetaY_fractionCut : " << std::setw(4) << getFractionCut(it->first, theta_Y) << std::setw(5) << "\n";
    } else {
      ss << "\n";
    }

    ss << "- Zcut             : " << std::setw(4) << (it->second).getZcut() << std::setw(5) << "   um"
       << "| sigZcut          : " << std::setw(4) << (it->second).getSigZcut() << std::setw(1) << " "
       << "| maxMoveZcut      : " << std::setw(4) << (it->second).getMaxMoveZcut() << std::setw(5) << "   um"
       << "| ErrorZcut        : " << std::setw(4) << (it->second).getErrorZcut() << std::setw(5) << "   um";

    if (floatMap_.find(it->first) != floatMap_.end()) {
      ss << "| Z_fractionCut      : " << std::setw(4) << getFractionCut(it->first, Z) << std::setw(5) << "\n";
    } else {
      ss << "\n";
    }

    ss << "- thetaZcut        : " << std::setw(4) << (it->second).getThetaZcut() << std::setw(5) << " urad"
       << "| sigThetaZcut     : " << std::setw(4) << (it->second).getSigThetaZcut() << std::setw(1) << " "
       << "| maxMoveThetaZcut : " << std::setw(4) << (it->second).getMaxMoveThetaZcut() << std::setw(5) << " urad"
       << "| ErrorThetaZcut   : " << std::setw(4) << (it->second).getErrorThetaZcut() << std::setw(5) << " urad";

    if (floatMap_.find(it->first) != floatMap_.end()) {
      ss << "| thetaZ_fractionCut : " << std::setw(4) << getFractionCut(it->first, theta_Z) << std::setw(5) << "\n";
    } else {
      ss << "\n";
    }

    out << ss.str() << std::endl;

    if ((it->second).hasExtraDOF()) {
      for (unsigned int j = 0; j < (it->second).extraDOFSize(); j++) {
        std::array<float, 4> extraDOFCuts = getExtraDOFCutsForAlignable(it->first, j);

        out << "Extra DOF " << j << " with label: " << getExtraDOFLabelForAlignable(it->first, j);
        out << "- cut              : " << std::setw(4) << extraDOFCuts.at(0) << std::setw(5) << "    "
            << "| sigCut           : " << std::setw(4) << extraDOFCuts.at(1) << std::setw(1) << " "
            << "| maxMoveCut       : " << std::setw(4) << extraDOFCuts.at(2) << std::setw(5) << "    "
            << "| maxErrorCut      : " << std::setw(4) << extraDOFCuts.at(3) << std::setw(5) << "    ";
      }
    }
  }
}
