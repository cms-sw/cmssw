#ifndef JetAlgorithms_CMSIterativeConeAlgorithm_h
#define JetAlgorithms_CMSIterativeConeAlgorithm_h

/** \class CMSIterativeConeAlgorithm
 *
 * CMSIterativeConeAlgorithm - iterative cone algorithm without 
 * jet merging/splitting. Originally implemented in ORCA by H.P.Wellish.  
 * Documented in CMS NOTE-2006/036
 *
 * \author A.Ulyanov, ITEP
 * $Id: CMSIterativeConeAlgorithm.h,v 1.1 2009/08/24 14:35:59 srappocc Exp $
 ************************************************************/


#include <vector>

#include "RecoParticleFlow/PFRootEvent/interface/JetRecoTypes.h"

class CMSIterativeConeAlgorithm{
 public:
  /** Constructor
  \param seed defines the minimum ET in GeV of a tower that can seed a jet.
  \param radius defines the maximum radius of a jet in eta-phi space.
  */
  CMSIterativeConeAlgorithm(double seed, double radius): 
    theSeedThreshold(seed),
    theConeRadius(radius)
    { }

  /// Find the ProtoJets from the collection of input Candidates.
  void run(const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput) const;

 private:

  double theSeedThreshold;
  double theConeRadius;
};

#endif
