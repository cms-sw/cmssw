#ifndef HEAVYIONSANALYSIS_EGMANALYSIS_ENERGYSCALECORRECTOR_H
#define HEAVYIONSANALYSIS_EGMANALYSIS_ENERGYSCALECORRECTOR_H

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/EgammaTools/interface/EpCombinationTool.h"

#include "TRandom.h"

#include <string>
#include <vector>

class EnergyScaleCorrector {
public:
  EnergyScaleCorrector(std::string const& file, EpCombinationTool const& combinator, TRandom* rng, float min_pt);

  ~EnergyScaleCorrector() = default;

  void calibrate(reco::GsfElectron& ele, int hiBin) const;

private:
  template <typename T>
  void readline(std::ifstream& f, std::vector<T>& data);

  void read(std::string const& file);

  std::pair<float, float> combined_momentum(reco::GsfElectron& ele, float scale, float smear) const;

  int index_for(int value) const;
  float scale_for(int hiBin, float eta) const;
  float smear_for(int hiBin, float eta) const;

  EpCombinationTool const* combinator_;
  TRandom* rng_;
  float min_pt_;

  std::vector<int> edges;
  std::vector<float> eb_scale;
  std::vector<float> ee_scale;
  std::vector<float> eb_smear;
  std::vector<float> ee_smear;
};

#endif /* HEAVYIONSANALYSIS_PHOTONANALYSIS_ENERGYSCALECORRECTOR_H */
