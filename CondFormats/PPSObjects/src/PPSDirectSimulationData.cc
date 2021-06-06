#include "CondFormats/PPSObjects/interface/PPSDirectSimulationData.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FileInPath.h"

#include "TFile.h"

#include <iostream>
#include <string>

PPSDirectSimulationData::PPSDirectSimulationData()
    : empiricalAperture45_(""),
      empiricalAperture56_(""),

      timeResolutionDiamonds45_(""),
      timeResolutionDiamonds56_("") {}

PPSDirectSimulationData::~PPSDirectSimulationData() {}

// Getters
const std::string &PPSDirectSimulationData::getEmpiricalAperture45() const { return empiricalAperture45_; }
const std::string &PPSDirectSimulationData::getEmpiricalAperture56() const { return empiricalAperture56_; }

const std::string &PPSDirectSimulationData::getTimeResolutionDiamonds45() const { return timeResolutionDiamonds45_; }
const std::string &PPSDirectSimulationData::getTimeResolutionDiamonds56() const { return timeResolutionDiamonds56_; }

std::map<unsigned int, PPSDirectSimulationData::FileObject> &PPSDirectSimulationData::getEfficienciesPerRP() {
  return efficienciesPerRP_;
}
std::map<unsigned int, PPSDirectSimulationData::FileObject> &PPSDirectSimulationData::getEfficienciesPerPlane() {
  return efficienciesPerPlane_;
};

// Setters
void PPSDirectSimulationData::setEmpiricalAperture45(std::string s) { empiricalAperture45_ = s; }
void PPSDirectSimulationData::setEmpiricalAperture56(std::string s) { empiricalAperture56_ = s; }

void PPSDirectSimulationData::setTimeResolutionDiamonds45(std::string s) { timeResolutionDiamonds45_ = s; }
void PPSDirectSimulationData::setTimeResolutionDiamonds56(std::string s) { timeResolutionDiamonds56_ = s; }

std::map<unsigned int, std::unique_ptr<TH2F>> PPSDirectSimulationData::loadEffeciencyHistogramsPerRP() const {
  std::map<unsigned int, std::unique_ptr<TH2F>> result;

  for (const auto &it : efficienciesPerRP_)
    result[it.first] = loadObject(it.second.first, it.second.second);

  return result;
}

std::map<unsigned int, std::unique_ptr<TH2F>> PPSDirectSimulationData::loadEffeciencyHistogramsPerPlane() const {
  std::map<unsigned int, std::unique_ptr<TH2F>> result;

  for (const auto &it : efficienciesPerPlane_) {
    CTPPSDetId rpId(it.first);

    if (rpId.subdetId() == CTPPSDetId::sdTrackingStrip) {
      for (unsigned int pl = 0; pl < 10; ++pl) {
        TotemRPDetId plId(rpId.arm(), rpId.station(), rpId.rp(), pl);
        result[plId] = loadObject(it.second.first, replace(it.second.second, "<detid>", std::to_string(pl)));
      }
    }

    if (rpId.subdetId() == CTPPSDetId::sdTrackingPixel) {
      for (unsigned int pl = 0; pl < 6; ++pl) {
        CTPPSPixelDetId plId(rpId.arm(), rpId.station(), rpId.rp(), pl);
        result[plId] = loadObject(it.second.first, replace(it.second.second, "<detid>", std::to_string(pl)));
      }
    }

    if (rpId.subdetId() == CTPPSDetId::sdTimingDiamond) {
      for (unsigned int pl = 0; pl < 4; ++pl) {
        CTPPSDiamondDetId plId(rpId.arm(), rpId.station(), rpId.rp(), pl);
        result[plId] = loadObject(it.second.first, replace(it.second.second, "<detid>", std::to_string(pl)));
      }
    }
  }

  return result;
}

std::unique_ptr<TH2F> PPSDirectSimulationData::loadObject(const std::string &file, const std::string &object) {
  edm::FileInPath fip(file.c_str());
  TFile *f_in = TFile::Open(fip.fullPath().c_str());
  if (!f_in)
    throw cms::Exception("PPS") << "Cannot open file '" << fip.fullPath() << "'.";

  TH2F *o_in = (TH2F *)f_in->Get(object.c_str());
  if (!o_in)
    throw cms::Exception("PPS") << "Cannot load object '" << object << "' from file '" << fip.fullPath() << "'.";

  // disassociate histogram from the file
  o_in->SetDirectory(nullptr);

  delete f_in;

  return std::unique_ptr<TH2F>(o_in);
}

std::string PPSDirectSimulationData::replace(std::string input, const std::string &from, const std::string &to) {
  size_t start_pos = 0;
  while ((start_pos = input.find(from, start_pos)) != std::string::npos) {
    input.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
  return input;
}
