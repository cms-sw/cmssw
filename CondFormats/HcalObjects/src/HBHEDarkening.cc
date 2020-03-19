#include "CondFormats/HcalObjects/interface/HBHEDarkening.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <cassert>

HBHEDarkening::HBHEDarkening(int ieta_shift,
                             float drdA,
                             float drdB,
                             const std::map<int, std::vector<std::vector<float>>>& dosemaps,
                             const std::vector<LumiYear>& years)
    : ieta_shift_(ieta_shift), drdA_(drdA), drdB_(drdB), dosemaps_(dosemaps), years_(years) {
  //finish initializing years
  std::sort(years_.begin(), years_.end());
  //sum up int lumi
  float sumlumi = 0.0;
  for (auto& year : years_) {
    sumlumi += year.intlumi_;
    year.sumlumi_ = sumlumi;
  }
}

std::vector<std::vector<float>> HBHEDarkening::readDoseMap(const std::string& fullpath) {
  std::ifstream infile(fullpath.c_str());
  if (!infile.is_open()) {
    throw cms::Exception("FileNotFound") << "Unable to open '" << fullpath << "'" << std::endl;
  }
  std::string line;
  std::vector<std::vector<float>> result;
  while (getline(infile, line)) {
    //space-separated
    std::stringstream linestream(line);
    std::vector<float> lineresult;
    float doseval;
    while (linestream >> doseval)
      lineresult.push_back(doseval);
    result.push_back(lineresult);
  }
  return result;
}

float HBHEDarkening::dose(int ieta, int lay, int energy) const {
  //existence check
  const auto dosemapIt = dosemaps_.find(energy);
  if (dosemapIt == dosemaps_.end())
    return 0.0;

  //bounds check
  const auto& dosemap = dosemapIt->second;
  if (ieta < 0 or ieta >= int(dosemap.size()))
    return 0.0;

  //bounds check
  const auto& doserow = dosemap[ieta];
  if (lay < 0 or lay >= int(doserow.size()))
    return 0.0;

  return doserow[lay];
}

std::string HBHEDarkening::getYearForLumi(float intlumi) const {
  //compare based on sum lumi value
  auto lb = std::lower_bound(years_.begin(), years_.end(), intlumi, LumiYearComp());
  if (lb == years_.end() or lb->sumlumi_ < intlumi) {
    throw cms::Exception("ValueError") << "HBHEDarkening: insufficient LHC run information provided to simulate "
                                       << intlumi << "/fb - check the python config" << std::endl;
  }
  return lb->year_;
}

float HBHEDarkening::degradationYear(const LumiYear& year, float intlumi, int ieta, int lay) const {
  float doseToUse = dose(ieta, lay, year.energy_);
  if (doseToUse == 0.0)
    return 1.0;

  //apply dose rate dependence model to the provided year
  //get krad/hr from Mrad/fb-1 and fb-1/hr
  float decayConst = drdA_ * std::pow(1000 * doseToUse * year.lumirate_, drdB_);

  //determine if this is a partial year
  float intlumiToUse = year.intlumi_;
  if (intlumi < year.sumlumi_)
    intlumiToUse = intlumi - (year.sumlumi_ - year.intlumi_);

  //calculate degradation
  return std::exp(-(intlumiToUse * doseToUse) / decayConst);
}

float HBHEDarkening::degradation(float intlumi, int ieta, int lay) const {
  ieta = abs(ieta);
  //shift ieta tower index to act as array index
  ieta -= ieta_shift_;
  //shift layer index by 1 to act as array index
  lay -= 1;

  //accumulate degradation over years
  float response = 1.0;
  std::string yearForLumi = getYearForLumi(intlumi);
  assert(yearForLumi.size());

  for (const auto& year : years_) {
    response *= degradationYear(year, intlumi, ieta, lay);
    if (year.year_ == yearForLumi)
      break;
  }

  return response;
}
