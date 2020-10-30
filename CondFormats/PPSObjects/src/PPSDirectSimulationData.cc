#include "CondFormats/PPSObjects/interface/PPSDirectSimulationData.h"
#include <iostream>

// Constructors
PPSDirectSimulationData::PPSDirectSimulationData():
    useEmpiricalApertures(0),
    empiricalAperture45(""),
    empiricalAperture56(""),
    timeResolutionDiamonds45(""),
    timeResolutionDiamonds56(""){}

// Destructor
PPSDirectSimulationData::~PPSDirectSimulationData() {}

// Getters
bool PPSDirectSimulationData::getUseEmpiricalApertures() const{return useEmpiricalApertures;}
const std::string& PPSDirectSimulationData::getEmpiricalAperture45() const{return empiricalAperture45;}
const std::string& PPSDirectSimulationData::getEmpiricalAperture56() const{return empiricalAperture56;}
const std::string& PPSDirectSimulationData::getTimeResolutionDiamonds45() const{return timeResolutionDiamonds45;}
const std::string& PPSDirectSimulationData::getTimeResolutionDiamonds56() const{return timeResolutionDiamonds56;}
bool PPSDirectSimulationData::getUseTimeEfficiencyCheck() const{return useTimeEfficiencyCheck;}
const std::string& PPSDirectSimulationData::getEffTimePath() const{return effTimePath;}
const std::string& PPSDirectSimulationData::getEffTimeObject45() const{return effTimeObject45;}
const std::string& PPSDirectSimulationData::getEffTimeObject56() const{return effTimeObject56;}


// Setters
void PPSDirectSimulationData::setUseEmpiricalApertures(bool b){useEmpiricalApertures=b;}
void PPSDirectSimulationData::setEmpiricalAperture45(std::string s){empiricalAperture45=s;}
void PPSDirectSimulationData::setEmpiricalAperture56(std::string s){empiricalAperture56=s;}
void PPSDirectSimulationData::setTimeResolutionDiamonds45(std::string s){timeResolutionDiamonds45=s;}
void PPSDirectSimulationData::setTimeResolutionDiamonds56(std::string s){timeResolutionDiamonds56=s;}
void PPSDirectSimulationData::setUseTimeEfficiencyCheck(bool b){useTimeEfficiencyCheck=b;}
void PPSDirectSimulationData::setEffTimePath(std::string s){effTimePath=s;}
void PPSDirectSimulationData::setEffTimeObject45(std::string s){effTimeObject45=s;}
void PPSDirectSimulationData::setEffTimeObject56(std::string s){effTimeObject56=s;}



void PPSDirectSimulationData::printInfo(std::stringstream& s) {
    s << "\n   useEmpiricalApertures = " << useEmpiricalApertures
    << "\n   empiricalAperture45 = " << empiricalAperture45
    << "\n   empiricalAperture56 = " << empiricalAperture56
    << "\n   timeResolutionDiamonds45 = " << timeResolutionDiamonds45
    << "\n   timeResolutionDiamonds56 = " << timeResolutionDiamonds56
    << "\n useTimeEfficiencyCheck= "<< useTimeEfficiencyCheck
    << "\n effTimePath= "<< effTimePath
    << "\n effTimeObject45= "<< effTimeObject45
    << "\n effTimeObject56= "<< effTimeObject56
    << std::endl;
}

std::ostream& operator<<(std::ostream& os, PPSDirectSimulationData info) {
  std::stringstream ss;
  info.printInfo(ss);
  os << ss.str();
  return os;
}