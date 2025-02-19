#ifndef __ReflectedIterator_h_
#define __ReflectedIterator_h_

#include "RecoJets/JetProducers/interface/PileUpSubtractor.h"

class ReflectedIterator : public PileUpSubtractor {
 public:
   ReflectedIterator(const edm::ParameterSet& iConfig) : PileUpSubtractor(iConfig),
     sumRecHits_(iConfig.getParameter<bool>("sumRecHits")),
     dropZeroTowers_(iConfig.getUntrackedParameter<bool>("dropZeroTowers",true))
       {;}
    virtual void offsetCorrectJets();
    void rescaleRMS(double s);
    double getEt(const reco::CandidatePtr & in) const;
    double getEta(const reco::CandidatePtr & in) const;
    virtual void calculatePedestal(std::vector<fastjet::PseudoJet> const & coll);
    virtual void subtractPedestal(std::vector<fastjet::PseudoJet> & coll);

    bool sumRecHits_;
    bool dropZeroTowers_;
    ~ReflectedIterator(){;}
    
};


#endif
