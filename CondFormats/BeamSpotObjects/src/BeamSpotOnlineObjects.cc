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
int BeamSpotOnlineObjects::numTracks() const { return BeamSpotOnlineObjectsImpl::getOneParam(intParams_, NUM_TRACKS); }

int BeamSpotOnlineObjects::numPVs() const { return BeamSpotOnlineObjectsImpl::getOneParam(intParams_, NUM_PVS); }

int BeamSpotOnlineObjects::usedEvents() const {
  return BeamSpotOnlineObjectsImpl::getOneParam(intParams_, USED_EVENTS);
}

int BeamSpotOnlineObjects::maxPVs() const { return BeamSpotOnlineObjectsImpl::getOneParam(intParams_, MAX_PVS); }

float BeamSpotOnlineObjects::meanPV() const { return BeamSpotOnlineObjectsImpl::getOneParam(floatParams_, MEAN_PV); }

float BeamSpotOnlineObjects::meanErrorPV() const {
  return BeamSpotOnlineObjectsImpl::getOneParam(floatParams_, ERR_MEAN_PV);
}

float BeamSpotOnlineObjects::rmsPV() const { return BeamSpotOnlineObjectsImpl::getOneParam(floatParams_, RMS_PV); }

float BeamSpotOnlineObjects::rmsErrorPV() const {
  return BeamSpotOnlineObjectsImpl::getOneParam(floatParams_, ERR_RMS_PV);
}

std::string BeamSpotOnlineObjects::startTime() const {
  return BeamSpotOnlineObjectsImpl::getOneParam(stringParams_, START_TIME);
}

std::string BeamSpotOnlineObjects::endTime() const {
  return BeamSpotOnlineObjectsImpl::getOneParam(stringParams_, END_TIME);
}

std::string BeamSpotOnlineObjects::lumiRange() const {
  return BeamSpotOnlineObjectsImpl::getOneParam(stringParams_, LUMI_RANGE);
}

cond::Time_t BeamSpotOnlineObjects::creationTime() const {
  return BeamSpotOnlineObjectsImpl::getOneParam(timeParams_, CREATE_TIME);
}

cond::Time_t BeamSpotOnlineObjects::startTimeStamp() const {
  return BeamSpotOnlineObjectsImpl::getOneParam(timeParams_, START_TIMESTAMP);
}

cond::Time_t BeamSpotOnlineObjects::endTimeStamp() const {
  return BeamSpotOnlineObjectsImpl::getOneParam(timeParams_, END_TIMESTAMP);
}

// setters
void BeamSpotOnlineObjects::copyFromBeamSpotObject(const BeamSpotObjects& bs) {
  setType(bs.beamType());
  setPosition(bs.x(), bs.y(), bs.z());
  setSigmaZ(bs.sigmaZ());
  setdxdz(bs.dxdz());
  setdydz(bs.dydz());
  setBeamWidthX(bs.beamWidthX());
  setBeamWidthY(bs.beamWidthY());
  setBeamWidthXError(bs.beamWidthXError());
  setBeamWidthYError(bs.beamWidthYError());
  setEmittanceX(bs.emittanceX());
  setEmittanceY(bs.emittanceY());
  setBetaStar(bs.betaStar());

  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      setCovariance(i, j, bs.covariance(i, j));
    }
  }
}

void BeamSpotOnlineObjects::setNumTracks(int nTracks) {
  BeamSpotOnlineObjectsImpl::setOneParam(intParams_, NUM_TRACKS, nTracks);
}

void BeamSpotOnlineObjects::setNumPVs(int nPVs) { BeamSpotOnlineObjectsImpl::setOneParam(intParams_, NUM_PVS, nPVs); }

void BeamSpotOnlineObjects::setUsedEvents(int usedEvents) {
  BeamSpotOnlineObjectsImpl::setOneParam(intParams_, USED_EVENTS, usedEvents);
}

void BeamSpotOnlineObjects::setMaxPVs(int maxPVs) {
  BeamSpotOnlineObjectsImpl::setOneParam(intParams_, MAX_PVS, maxPVs);
}

void BeamSpotOnlineObjects::setMeanPV(float meanPVs) {
  BeamSpotOnlineObjectsImpl::setOneParam(floatParams_, MEAN_PV, meanPVs);
}

void BeamSpotOnlineObjects::setMeanErrorPV(float errMeanPVs) {
  BeamSpotOnlineObjectsImpl::setOneParam(floatParams_, ERR_MEAN_PV, errMeanPVs);
}

void BeamSpotOnlineObjects::setRmsPV(float rmsPVs) {
  BeamSpotOnlineObjectsImpl::setOneParam(floatParams_, RMS_PV, rmsPVs);
}

void BeamSpotOnlineObjects::setRmsErrorPV(float errRmsPVs) {
  BeamSpotOnlineObjectsImpl::setOneParam(floatParams_, ERR_RMS_PV, errRmsPVs);
}

void BeamSpotOnlineObjects::setStartTime(std::string startTime) {
  BeamSpotOnlineObjectsImpl::setOneParam(stringParams_, START_TIME, startTime);
}

void BeamSpotOnlineObjects::setEndTime(std::string endTime) {
  BeamSpotOnlineObjectsImpl::setOneParam(stringParams_, END_TIME, endTime);
}

void BeamSpotOnlineObjects::setLumiRange(std::string lumiRange) {
  BeamSpotOnlineObjectsImpl::setOneParam(stringParams_, LUMI_RANGE, lumiRange);
}

void BeamSpotOnlineObjects::setCreationTime(cond::Time_t createTime) {
  BeamSpotOnlineObjectsImpl::setOneParam(timeParams_, CREATE_TIME, createTime);
}

void BeamSpotOnlineObjects::setStartTimeStamp(cond::Time_t starTimeStamp) {
  BeamSpotOnlineObjectsImpl::setOneParam(timeParams_, START_TIMESTAMP, starTimeStamp);
}

void BeamSpotOnlineObjects::setEndTimeStamp(cond::Time_t endTimeStamp) {
  BeamSpotOnlineObjectsImpl::setOneParam(timeParams_, END_TIMESTAMP, endTimeStamp);
}

// printers
void BeamSpotOnlineObjects::print(std::stringstream& ss) const {
  ss << "-----------------------------------------------------\n"
     << "              BeamSpotOnline Data\n\n"
     << " Beam type    = " << beamType() << "\n"
     << "       X0     = " << x() << " +/- " << xError() << " [cm]\n"
     << "       Y0     = " << y() << " +/- " << yError() << " [cm]\n"
     << "       Z0     = " << z() << " +/- " << zError() << " [cm]\n"
     << " Sigma Z0     = " << sigmaZ() << " +/- " << sigmaZError() << " [cm]\n"
     << " dxdz         = " << dxdz() << " +/- " << dxdzError() << " [radians]\n"
     << " dydz         = " << dydz() << " +/- " << dydzError() << " [radians]\n"
     << " Beam Width X = " << beamWidthX() << " +/- " << beamWidthXError() << " [cm]\n"
     << " Beam Width Y = " << beamWidthY() << " +/- " << beamWidthYError() << " [cm]\n"
     << " Emittance X  = " << emittanceX() << " [cm]\n"
     << " Emittance Y  = " << emittanceY() << " [cm]\n"
     << " Beta star    = " << betaStar() << " [cm]\n"
     << " Last Lumi    = " << lastAnalyzedLumi() << "\n"
     << " Last Run     = " << lastAnalyzedRun() << "\n"
     << " Last Fill    = " << lastAnalyzedFill() << "\n"
     << "-----------------------------------------------------\n\n";
}

std::ostream& operator<<(std::ostream& os, BeamSpotOnlineObjects beam) {
  std::stringstream ss;
  beam.print(ss);
  os << ss.str();
  return os;
}
