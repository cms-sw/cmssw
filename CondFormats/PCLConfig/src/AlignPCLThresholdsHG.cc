#include "CondFormats/PCLConfig/interface/AlignPCLThresholdsHG.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <iomanip>  // std::setw

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

const std::vector<float> &AlignPCLThresholdsHG::getFloatVec(const std::string &AlignableId) const {
  std::unordered_map<std::string, std::vector<float>>::const_iterator it = floatMap.find(AlignableId);

  if (it != floatMap.end()) {
    return it->second;
  } else {
    throw cms::Exception("AlignPCLThresholdsHG") << "No float vector defined for Alignable id " << AlignableId << "\n";
  }
}

void AlignPCLThresholdsHG::SetFractionCut(const std::string &AlignableId, const coordType &type, const float &cut) {
  // Set entry in map if not yet available
  std::unordered_map<std::string, std::vector<float>>::const_iterator it = floatMap.find(AlignableId);
  if (it == floatMap.end())
    floatMap[AlignableId] = std::vector<float>(FSIZE, 0.);

  switch (type) {
    case X:
      return AlignPCLThresholdsHGImpl::setParam(floatMap[AlignableId], FRACTION_CUT_X, cut);
    case Y:
      return AlignPCLThresholdsHGImpl::setParam(floatMap[AlignableId], FRACTION_CUT_Y, cut);
    case Z:
      return AlignPCLThresholdsHGImpl::setParam(floatMap[AlignableId], FRACTION_CUT_Z, cut);
    case theta_X:
      return AlignPCLThresholdsHGImpl::setParam(floatMap[AlignableId], FRACTION_CUT_TX, cut);
    case theta_Y:
      return AlignPCLThresholdsHGImpl::setParam(floatMap[AlignableId], FRACTION_CUT_TY, cut);
    case theta_Z:
      return AlignPCLThresholdsHGImpl::setParam(floatMap[AlignableId], FRACTION_CUT_TZ, cut);
    default:
      throw cms::Exception("AlignPCLThresholdsHG")
          << "Requested setting fraction threshold for undefined coordinate" << type << "\n";
  }
}

std::array<float, 6> AlignPCLThresholdsHG::getFractionCut(const std::string &AlignableId) const {
  const std::vector<float> vec = getFloatVec(AlignableId);
  return {{AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_X),
           AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_Y),
           AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_Z),
           AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_TX),
           AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_TY),
           AlignPCLThresholdsHGImpl::getParam(vec, FRACTION_CUT_TZ)}};
}

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

int AlignPCLThresholdsHG::payloadVersion() const {
  switch (FSIZE) {
    case 6:
      return 1;
    default:
      throw cms::Exception("AlignPCLThresholdsHG")
          << "Payload version with FSIZE equal to " << FSIZE << " is not defined.\n";
  }
}

void AlignPCLThresholdsHG::printAllHG() const {
  edm::LogVerbatim("AlignPCLThresholdsHG") << "AlignPCLThresholdsHG::printAllHG()";
  edm::LogVerbatim("AlignPCLThresholdsHG") << " ==================================";
  for (auto it = floatMap.begin(); it != floatMap.end(); ++it) {
    edm::LogVerbatim("AlignPCLThresholdsHG") << " ==================================";
    edm::LogVerbatim("AlignPCLThresholdsHG")
        << "key : " << it->first << " \n"
        << "- X_fractionCut             : " << std::setw(4) << getFractionCut(it->first, X) << std::setw(5) << "\n"

        << "- thetaX_fractionCut        : " << std::setw(4) << getFractionCut(it->first, theta_X) << std::setw(5)
        << "\n"

        << "- Y_fractionCut             : " << std::setw(4) << getFractionCut(it->first, Y) << std::setw(5) << "\n"

        << "- thetaY_fractionCut        : " << std::setw(4) << getFractionCut(it->first, theta_Y) << std::setw(5)
        << "\n"

        << "- Z_fractionCut             : " << std::setw(4) << getFractionCut(it->first, Z) << std::setw(5) << "\n"

        << "- thetaZ_fractionCut        : " << std::setw(4) << getFractionCut(it->first, theta_Z) << std::setw(5);
  }
}
