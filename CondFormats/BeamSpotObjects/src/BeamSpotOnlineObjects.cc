#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"

#include <iostream>

namespace BeamSpotOnlineObjectsImpl {
  template <typename T>
  const T& getParams(const std::vector<T>& params, size_t index) {
    if (index >= params.size())
      throw std::out_of_range("Parameter with index " + std::to_string(index) + " is out of range.");
    return params[index];
  }

  template <typename T>
  T& accessParams(std::vector<T>& params, size_t index) {
    if (index >= params.size())
      throw std::out_of_range("Parameter with index " + std::to_string(index) + " is out of range.");
    return params[index];
  }

  template <typename T>
  const T& getOneParam(const std::vector<std::vector<T> >& params, size_t index) {
    if (index >= params.size())
      throw std::out_of_range("Parameter with index " + std::to_string(index) + " is out of range.");
    const std::vector<T>& inner = params[index];
    if (inner.empty())
      throw std::out_of_range("Parameter with index " + std::to_string(index) + " type=" + typeid(T).name() +
                              " has no value stored.");
    return inner[0];
  }

  template <typename T>
  void setOneParam(std::vector<std::vector<T> >& params, size_t index, const T& value) {
    if (index >= params.size())
      throw std::out_of_range("Parameter with index " + std::to_string(index) + " is out of range.");
    params[index] = std::vector<T>(1, value);
  }

  template <typename T>
  void setParams(std::vector<T>& params, size_t index, const T& value) {
    if (index >= params.size())
      throw std::out_of_range("Parameter with index " + std::to_string(index) + " is out of range.");
    params[index] = value;
  }

}  //namespace BeamSpotOnlineObjectsImpl

// getters
int BeamSpotOnlineObjects::GetNumTracks() const {
  return BeamSpotOnlineObjectsImpl::getOneParam(intParams_, NUM_TRACKS);
}

int BeamSpotOnlineObjects::GetNumPVs() const { return BeamSpotOnlineObjectsImpl::getOneParam(intParams_, NUM_PVS); }

cond::Time_t BeamSpotOnlineObjects::GetCreationTime() const {
  return BeamSpotOnlineObjectsImpl::getOneParam(timeParams_, CREATE_TIME);
}

// setters
void BeamSpotOnlineObjects::SetNumTracks(int nTracks) {
  BeamSpotOnlineObjectsImpl::setOneParam(intParams_, NUM_TRACKS, nTracks);
}

void BeamSpotOnlineObjects::SetNumPVs(int nPVs) { BeamSpotOnlineObjectsImpl::setOneParam(intParams_, NUM_PVS, nPVs); }

void BeamSpotOnlineObjects::SetCreationTime(cond::Time_t createTime) {
  BeamSpotOnlineObjectsImpl::setOneParam(timeParams_, CREATE_TIME, createTime);
}

// printers
void BeamSpotOnlineObjects::print(std::stringstream& ss) const {
  ss << "-----------------------------------------------------\n"
     << "              BeamSpotOnline Data\n\n"
     << " Beam type    = " << GetBeamType() << "\n"
     << "       X0     = " << GetX() << " +/- " << GetXError() << " [cm]\n"
     << "       Y0     = " << GetY() << " +/- " << GetYError() << " [cm]\n"
     << "       Z0     = " << GetZ() << " +/- " << GetZError() << " [cm]\n"
     << " Sigma Z0     = " << GetSigmaZ() << " +/- " << GetSigmaZError() << " [cm]\n"
     << " dxdz         = " << Getdxdz() << " +/- " << GetdxdzError() << " [radians]\n"
     << " dydz         = " << Getdydz() << " +/- " << GetdydzError() << " [radians]\n"
     << " Beam Width X = " << GetBeamWidthX() << " +/- " << GetBeamWidthXError() << " [cm]\n"
     << " Beam Width Y = " << GetBeamWidthY() << " +/- " << GetBeamWidthYError() << " [cm]\n"
     << " Emittance X  = " << GetEmittanceX() << " [cm]\n"
     << " Emittance Y  = " << GetEmittanceY() << " [cm]\n"
     << " Beta star    = " << GetBetaStar() << " [cm]\n"
     << " Last Lumi    = " << GetLastAnalyzedLumi() << "\n"
     << " Last Run     = " << GetLastAnalyzedRun() << "\n"
     << " Last Fill    = " << GetLastAnalyzedFill() << "\n"
     << "-----------------------------------------------------\n\n";
}

std::ostream& operator<<(std::ostream& os, BeamSpotOnlineObjects beam) {
  std::stringstream ss;
  beam.print(ss);
  os << ss.str();
  return os;
}