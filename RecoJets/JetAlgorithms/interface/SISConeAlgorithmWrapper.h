
#ifndef JetAlgorithms_SISConeAlgorithmWrapper_h
#define JetAlgorithms_SISConeAlgorithmWrapper_h

/**
 * Interface to Seedless Infrared Safe Cone algorithm (http://projects.hepforge.org/siscone)
 * F.Ratnikov, UMd, June 22, 2007
 **/

#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"

namespace fastjet {
  class SISConePlugin;
}

class SISConeAlgorithmWrapper
{
 public:
  typedef std::vector<ProtoJet*> InternalCollection;

  
  SISConeAlgorithmWrapper ();

  /// Constructor takes as input all the values of the algorithm that the user can change, and the CaloTower Collection pointer.
  SISConeAlgorithmWrapper(
			  double fConeRadius, 
			  double fConeOverlapThreshold = 0.5,
			  const std::string& fSplitMergeScale = "pttilde",
			  int fMaxPasses = 0, 
			  double fProtojetPtMin = 0.,
			  bool fCaching = false,
			  int fDebug = 0  // not used
			  );

  ~SISConeAlgorithmWrapper ();

  /// Runs the algorithm and returns a list of caloJets. 
  /// The user declares the vector and calls this method.
  void run(const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput);


 private:
  fastjet::SISConePlugin* mPlugin;
}; 

#endif
