#ifndef JetAlgorithms_CDFMidpointAlgorithmWrapper_h
#define JetAlgorithms_CDFMidpointAlgorithmWrapper_h


#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"

namespace fastjet {
  class CDFMidPointPlugin;
}

class CDFMidpointAlgorithmWrapper
{
 public:
  typedef std::vector<ProtoJet*> InternalCollection;

  
  CDFMidpointAlgorithmWrapper ();

  /// Constructor takes as input all the values of the algorithm that the user can change, and the CaloTower Collection pointer.
  CDFMidpointAlgorithmWrapper(double fSeedThreshold, 
			      double fConeRadius, 
			      double fConeAreaFraction,
			      int fMaxPairSize, 
			      int fMaxIterations,
			      double fOverlapThreshold,
			      int fDebugLevel);

  ~CDFMidpointAlgorithmWrapper ();

  /// Runs the algorithm and returns a list of caloJets. 
  /// The user declares the vector and calls this method.
  void run(const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput);


 private:
  fastjet::CDFMidPointPlugin* mPlugin;
};

#endif
