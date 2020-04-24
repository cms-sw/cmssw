#ifndef __MultipleAlgoIterator_h_
#define __MultipleAlgoIterator_h_

#include "RecoJets/JetProducers/interface/PileUpSubtractor.h"

class MultipleAlgoIterator : public PileUpSubtractor {
 public:
 MultipleAlgoIterator(const edm::ParameterSet& iConfig, edm::ConsumesCollector && iC) : PileUpSubtractor(iConfig, std::move(iC)),
     sumRecHits_(iConfig.getParameter<bool>("sumRecHits")),
     dropZeroTowers_(iConfig.getUntrackedParameter<bool>("dropZeroTowers",true))
       {;}
    void offsetCorrectJets() override;
    void rescaleRMS(double s);
    double getEt(const reco::CandidatePtr & in) const;
    double getEta(const reco::CandidatePtr & in) const;
    void calculatePedestal(std::vector<fastjet::PseudoJet> const & coll) override;
    void subtractPedestal(std::vector<fastjet::PseudoJet> & coll) override;

    bool sumRecHits_;
    bool dropZeroTowers_;
    ~MultipleAlgoIterator() override{;}
    
};


#endif
