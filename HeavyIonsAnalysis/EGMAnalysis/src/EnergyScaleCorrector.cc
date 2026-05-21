#include "HeavyIonsAnalysis/EGMAnalysis/interface/EnergyScaleCorrector.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <fstream>
#include <iterator>

/* operator>> overload for std::vector<T> */
template <class T>
std::istream& operator>>(std::istream& s, std::vector<T>& v) {
  std::copy(std::istream_iterator<T>(s), std::istream_iterator<T>(), std::back_inserter(v));
  return s;
}

EnergyScaleCorrector::EnergyScaleCorrector(std::string const& file,
                                           EpCombinationTool const& combinator,
                                           TRandom* rng,
                                           float min_pt)
    : combinator_(&combinator), rng_(rng), min_pt_(min_pt) {
  read(file);
}

void EnergyScaleCorrector::calibrate(reco::GsfElectron& ele, int hiBin) const {
  /* skip low pt electrons */
  if (ele.pt() < min_pt_) {
    return;
  }

  auto scale = scale_for(hiBin, ele.superCluster()->eta());
  auto smear = smear_for(hiBin, ele.superCluster()->eta());

  auto fsmear = rng_->Gaus(1., smear / 91.1876);
  auto fscale = scale * fsmear;

  math::XYZTLorentzVector const oldP4 = ele.p4();
  auto combined = combined_momentum(ele, fscale, fsmear);

  float energy_scale = combined.first / oldP4.t();
  math::XYZTLorentzVector const newP4(
      oldP4.x() * energy_scale, oldP4.y() * energy_scale, oldP4.z() * energy_scale, combined.first);

  ele.correctMomentum(newP4, ele.trackMomentumError(), combined.second);
}

std::pair<float, float> EnergyScaleCorrector::combined_momentum(reco::GsfElectron& ele,
                                                                float scale,
                                                                float smear) const {
  const float oldEcalEnergy = ele.ecalEnergy();
  const float oldEcalEnergyErr = ele.ecalEnergyError();

  ele.setCorrectedEcalEnergy(oldEcalEnergy * scale);
  ele.setCorrectedEcalEnergyError(std::hypot(oldEcalEnergyErr * scale, oldEcalEnergy * smear * scale));

  return combinator_->combine(ele, oldEcalEnergyErr * scale);
}

float EnergyScaleCorrector::scale_for(int hiBin, float eta) const {
  int index = index_for(hiBin);
  return std::abs(eta) < 1.5 ? eb_scale[index] : ee_scale[index];
}

float EnergyScaleCorrector::smear_for(int hiBin, float eta) const {
  int index = index_for(hiBin);
  return std::abs(eta) < 1.5 ? eb_smear[index] : ee_smear[index];
}

template <typename T>
void EnergyScaleCorrector::readline(std::ifstream& f, std::vector<T>& data) {
  std::string line;
  std::getline(f, line);
  std::stringstream l(line);
  l >> data;
}

void EnergyScaleCorrector::read(std::string const& file) {
  std::ifstream f(edm::FileInPath(file).fullPath());
  if (!f.good())
    throw cms::Exception("EnergyScaleCorrector") << "bad input";

  readline(f, edges);
  readline(f, eb_scale);
  readline(f, ee_scale);
  readline(f, eb_smear);
  readline(f, ee_smear);

  f.close();
}

int EnergyScaleCorrector::index_for(int value) const {
  int index = edges.size();
  for (auto edge : edges)
    if (value < edge)
      --index;

  return index;
}
