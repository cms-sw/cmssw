#ifndef RecoTrackerDeDx_BTagLikeDeDxDiscriminator_h
#define RecoTrackerDeDx_BTagLikeDeDxDiscriminator_h

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"

class BTagLikeDeDxDiscriminator : public BaseDeDxEstimator {
public:
  BTagLikeDeDxDiscriminator(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iCollector)
      : token_(DeDxTools::esConsumes(iConfig.getParameter<std::string>("Reccord"), iCollector)) {
    meVperADCStrip =
        iConfig.getParameter<double>("MeVperADCStrip");  //currently needed until the map on the database are redone
    ProbabilityMode = iConfig.getParameter<std::string>("ProbabilityMode");
    Prob_ChargePath = nullptr;
  }

  void beginRun(edm::Run const& run, const edm::EventSetup& iSetup) override {
    auto const& histD3D = DeDxTools::getHistogramD3D(iSetup, token_);
    DeDxTools::buildDiscrimMap(histD3D, ProbabilityMode, Prob_ChargePath);
  }

  std::pair<float, float> dedx(const reco::DeDxHitCollection& Hits) override {
    std::vector<float> vect_probs;
    for (size_t i = 0; i < Hits.size(); i++) {
      float path = Hits[i].pathLength() * 10.0;  //x10 in order to be compatible with the map content
      float charge =
          Hits[i].charge() /
          (10.0 *
           meVperADCStrip);  // 10/meVperADCStrip in order to be compatible with the map content in ADC/mm instead of MeV/cm

      int BinX = Prob_ChargePath->GetXaxis()->FindBin(Hits[i].momentum());
      int BinY = Prob_ChargePath->GetYaxis()->FindBin(path);
      int BinZ = Prob_ChargePath->GetZaxis()->FindBin(charge);
      float prob = Prob_ChargePath->GetBinContent(BinX, BinY, BinZ);
      if (prob >= 0)
        vect_probs.push_back(prob);
    }

    size_t size = vect_probs.size();
    if (size <= 0)
      return std::make_pair(-1, -1);
    std::sort(vect_probs.begin(), vect_probs.end(), std::less<float>());
    float SumJet = 0.;
    for (size_t i = 0; i < size; i++) {
      if (vect_probs[i] <= 0.0001)
        vect_probs[i] = 0.0001;
      SumJet += log(vect_probs[i]);
    }

    float Loginvlog = log(-SumJet);
    float Prob = 1;
    float lfact = 1;

    for (size_t l = 1; l != size; l++) {
      lfact *= l;
      Prob += exp(l * Loginvlog - log(lfact));
    }

    float LogProb = log(Prob);
    float ProbJet = std::min((float)exp(std::max(LogProb + SumJet, -30.0f)), 1.0f);
    float TotalProb = -log10(ProbJet) / 4;
    TotalProb = 1 - TotalProb;

    return std::make_pair(TotalProb, -1);
  }

private:
  float meVperADCStrip;
  DeDxTools::ESGetTokenH3DDVariant token_;
  std::string ProbabilityMode;
  TH3F* Prob_ChargePath;
};

#endif
