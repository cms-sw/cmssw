#ifndef JetAlgorithms_FastJetBaseWrapper_h
#define JetAlgorithms_FastJetBaseWrapper_h


#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

namespace fastjet {
  class JetDefinition;
  class GhostedAreaSpec;
}

class FastJetBaseWrapper {
 public:
  FastJetBaseWrapper(const edm::ParameterSet& fConfig);
  virtual ~FastJetBaseWrapper();
  void run(const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput);
 protected:
  fastjet::JetDefinition* mJetDefinition;
 private:
  fastjet::GhostedAreaSpec* mActiveArea;
  double mJetPtMin;
}; 

#endif
