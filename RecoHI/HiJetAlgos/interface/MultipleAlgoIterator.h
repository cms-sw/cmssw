#ifndef __MultipleAlgoIterator_h_
#define __MultipleAlgoIterator_h_

#include "RecoJets/JetProducers/interface/PileUpSubtractor.h"

class MultipleAlgoIterator : public PileUpSubtractor {
 public:
  MultipleAlgoIterator(const edm::ParameterSet& iConfig) : PileUpSubtractor(iConfig) {;}
    virtual void offsetCorrectJets();
    ~MultipleAlgoIterator(){;}
    
};


#endif
