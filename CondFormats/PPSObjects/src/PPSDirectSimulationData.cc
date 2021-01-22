#include "CondFormats/PPSObjects/interface/PPSDirectSimulationData.h"
#include <iostream>

// Constructors
PPSDirectSimulationData::PPSDirectSimulationData()
    : empiricalAperture45_(""),
      empiricalAperture56_(""),

      timeResolutionDiamonds45_(""),
      timeResolutionDiamonds56_(""),

      effTimePath_(""),
      effTimeObject45_(""),
      effTimeObject56_("") {}

// Destructor
PPSDirectSimulationData::~PPSDirectSimulationData() {}

// Getters
const std::string& PPSDirectSimulationData::getEmpiricalAperture45() const { return empiricalAperture45_; }
const std::string& PPSDirectSimulationData::getEmpiricalAperture56() const { return empiricalAperture56_; }

const std::string& PPSDirectSimulationData::getTimeResolutionDiamonds45() const { return timeResolutionDiamonds45_; }
const std::string& PPSDirectSimulationData::getTimeResolutionDiamonds56() const { return timeResolutionDiamonds56_; }

const std::string& PPSDirectSimulationData::getEffTimePath() const { return effTimePath_; }
const std::string& PPSDirectSimulationData::getEffTimeObject45() const { return effTimeObject45_; }
const std::string& PPSDirectSimulationData::getEffTimeObject56() const { return effTimeObject56_; }

// Setters
void PPSDirectSimulationData::setEmpiricalAperture45(std::string s) { empiricalAperture45_ = s; }
void PPSDirectSimulationData::setEmpiricalAperture56(std::string s) { empiricalAperture56_ = s; }

void PPSDirectSimulationData::setTimeResolutionDiamonds45(std::string s) { timeResolutionDiamonds45_ = s; }
void PPSDirectSimulationData::setTimeResolutionDiamonds56(std::string s) { timeResolutionDiamonds56_ = s; }

void PPSDirectSimulationData::setEffTimePath(std::string s) { effTimePath_ = s; }
void PPSDirectSimulationData::setEffTimeObject45(std::string s) { effTimeObject45_ = s; }
void PPSDirectSimulationData::setEffTimeObject56(std::string s) { effTimeObject56_ = s; }

void PPSDirectSimulationData::printInfo(std::stringstream& s) {
  s << "\nempiricalAperture45 = " << empiricalAperture45_ << "\nempiricalAperture56 = " << empiricalAperture56_
    << "\ntimeResolutionDiamonds45 = " << timeResolutionDiamonds45_
    << "\ntimeResolutionDiamonds56 = " << timeResolutionDiamonds56_ << "\neffTimePath= " << effTimePath_
    << "\neffTimeObject45= " << effTimeObject45_ << "\neffTimeObject56= " << effTimeObject56_ << std::endl;
}

std::ostream& operator<<(std::ostream& os, PPSDirectSimulationData info) {
  std::stringstream ss;
  info.printInfo(ss);
  os << ss.str();
  return os;
}