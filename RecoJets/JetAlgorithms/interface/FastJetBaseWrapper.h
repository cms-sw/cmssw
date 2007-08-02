#ifndef JetAlgorithms_FastJetBaseWrapper_h
#define JetAlgorithms_FastJetBaseWrapper_h


#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"


namespace fastjet {
  class JetDefinition;
  class ActiveAreaSpec;
}

class FastJetBaseWrapper {
 public:
  FastJetBaseWrapper(const edm::ParameterSet& fConfig);
  virtual ~FastJetBaseWrapper();
  void run(const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput);
 protected:
  virtual void makeJetDefinition (const edm::ParameterSet& fConfig); // must be implemented in derived class
  fastjet::JetDefinition* mJetDefinition;
 private:
  void makeActiveArea (const edm::ParameterSet& fConfig);
  fastjet::ActiveAreaSpec* mActiveArea;
  double mPtMin;
  double mDCut;
  int mNJets;
}; 

#endif
