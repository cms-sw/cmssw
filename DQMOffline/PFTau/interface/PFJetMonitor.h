#ifndef DQMOffline_PFTau_PFJetMonitor_h
#define DQMOffline_PFTau_PFJetMonitor_h

#include "DQMOffline/PFTau/interface/Benchmark.h"
#include "DQMOffline/PFTau/interface/CandidateBenchmark.h"
#include "DQMOffline/PFTau/interface/MatchCandidateBenchmark.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/BasicJetCollection.h"

#include <vector>
#include <numeric>    // std::iota
#include <algorithm>  // std::sort

#include <TH1.h>  //needed by the deltaR->Fill() call

class PFJetMonitor : public Benchmark {
public:
  PFJetMonitor(float dRMax = 0.3, bool matchCharge = true, Benchmark::Mode mode = Benchmark::DEFAULT);

  ~PFJetMonitor() override;

  /// set the parameters accessing them from ParameterSet
  void setParameters(const edm::ParameterSet &parameterSet);

  /// set directory (to use in ROOT)
  void setDirectory(TDirectory *dir) override;

  /// book histograms
  void setup(DQMStore::IBooker &b);
  void setup(DQMStore::IBooker &b, const edm::ParameterSet &parameterSet);

  /// fill histograms with all particle
  template <class T, class C>
  void fill(const T &candidateCollection,
            const C &matchedCandCollection,
            float &minVal,
            float &maxVal,
            float &jetpT,
            const edm::ParameterSet &parameterSet);

  void fillOne(const reco::Jet &jet, const reco::Jet &matchedJet);

protected:
  CandidateBenchmark candBench_;
  MatchCandidateBenchmark matchCandBench_;

  TH2F *delta_frac_VS_frac_muon_;
  TH2F *delta_frac_VS_frac_photon_;
  TH2F *delta_frac_VS_frac_electron_;
  TH2F *delta_frac_VS_frac_charged_hadron_;
  TH2F *delta_frac_VS_frac_neutral_hadron_;

  TH1F *deltaR_;
  float dRMax_;
  bool onlyTwoJets_;
  bool matchCharge_;
  bool createPFractionHistos_;
  bool histogramBooked_;
};

#include "DQMOffline/PFTau/interface/Matchers.h"

template <class T, class C>
void PFJetMonitor::fill(const T &jetCollection,
                        const C &matchedJetCollection,
                        float &minVal,
                        float &maxVal,
                        float &jetpT,
                        const edm::ParameterSet &parameterSet) {
  std::vector<int> matchIndices;
  PFB::match(jetCollection, matchedJetCollection, matchIndices, matchCharge_, dRMax_);
  // now matchIndices[i] stores the j-th closest matched jet

  std::vector<uint32_t> sorted_pt_indices(jetCollection.size());
  std::iota(std::begin(sorted_pt_indices), std::end(sorted_pt_indices), 0);
  // Sort the vector of indices using the pt() as ordering variable
  std::sort(std::begin(sorted_pt_indices), std::end(sorted_pt_indices), [&](uint32_t i, uint32_t j) {
    return jetCollection[i].pt() < jetCollection[j].pt();
  });
  for (uint32_t i = 0; i < sorted_pt_indices.size(); ++i) {
    // If we want only the 2 pt-leading jets, now that they are orderd, simply
    // check if the index is either in the first or second location of the
    // sorted indices: if not, bail out.
    if (onlyTwoJets_ && i > 1)
      break;

    const reco::Jet &jet = jetCollection[i];

    if (!isInRange(jet.pt(), jet.eta(), jet.phi()))
      continue;

    int iMatch = matchIndices[i];
    assert(iMatch < static_cast<int>(matchedJetCollection.size()));

    if (iMatch != -1) {
      const reco::Jet &matchedJet = matchedJetCollection[iMatch];
      if (!isInRange(matchedJet.pt(), matchedJet.eta(), matchedJet.phi()))
        continue;

      float ptRes = (jet.pt() - matchedJet.pt()) / matchedJet.pt();

      jetpT = jet.pt();
      if (ptRes > maxVal)
        maxVal = ptRes;
      if (ptRes < minVal)
        minVal = ptRes;

      candBench_.fillOne(jet);  // fill pt eta phi and charge histos for MATCHED candidate jet
      matchCandBench_.fillOne(jet, matchedJet, parameterSet);  // fill delta_x_VS_y histos for matched couple
      if (createPFractionHistos_ && histogramBooked_)
        fillOne(jet, matchedJet);  // book and fill delta_frac_VS_frac histos for matched couple
    }

    for (unsigned j = 0; j < matchedJetCollection.size(); ++j)  // for DeltaR spectrum
      if (deltaR_)
        deltaR_->Fill(reco::deltaR(jetCollection[i], matchedJetCollection[j]));
  }  // end loop on jetCollection
}
#endif
