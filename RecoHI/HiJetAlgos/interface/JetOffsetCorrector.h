#ifndef __JetOffsetCorrector_h_
#define __JetOffsetCorrector_h_


#include "RecoJets/JetProducers/interface/PileUpSubtractor.h"

class JetOffsetCorrector : public PileUpSubtractor {
 public:
  JetOffsetCorrector(const edm::ParameterSet& iConfig) : PileUpSubtractor(iConfig) {;}
    ~JetOffsetCorrector(){;}
    
};


#endif
