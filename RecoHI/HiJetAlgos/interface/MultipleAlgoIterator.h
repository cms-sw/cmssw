#ifndef __MultipleAlgoIterator_h_
#define __MultipleAlgoIterator_h_

#include "RecoJets/JetProducers/interface/PileUpSubtractor.h"

class MultipleAlgoIterator : public PileUpSubtractor {
public:
  MultipleAlgoIterator(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
  void offsetCorrectJets() override;
  void rescaleRMS(double s);
  double getEt(const reco::CandidatePtr& in) const;
  double getEta(const reco::CandidatePtr& in) const;
  void calculatePedestal(std::vector<fastjet::PseudoJet> const& coll) override;
  void subtractPedestal(std::vector<fastjet::PseudoJet>& coll) override;
  void calculateOrphanInput(std::vector<fastjet::PseudoJet>& orphanInput) override;

private:
  double minimumTowersFraction_;
  bool sumRecHits_;
  bool dropZeroTowers_;
};

#endif
