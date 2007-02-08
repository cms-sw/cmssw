#ifndef JetAlgorithms_CMSIterativeConeAlgorithm_h
#define JetAlgorithms_CMSIterativeConeAlgorithm_h

/** \class CMSIterativeConeAlgorithm
 *
 * CMSIterativeConeAlgorithm - iterative cone algorithm without 
 * jet merging/splitting. Originally implemented in ORCA by H.P.Wellish.  
 * Documented in CMS NOTE-2006/036
 *
 * \author A.Ulyanov, ITEP
 * $Id: CMSIterativeConeAlgorithm.h,v 1.4 2006/12/05 18:37:43 fedor Exp $
 ************************************************************/


#include <vector>

#include "RecoJets/JetAlgorithms/interface/JetRecoTypes.h"

class CMSIterativeConeAlgorithm{
 public:
  /** Constructor
  \param seed defines the minimum ET in GeV of a tower that can seed a jet.
  \param radius defines the maximum radius of a jet in eta-phi space.
  \param towerThreshold defines the minimum ET in GeV for a tower to be inluded in a jet 
  */
  CMSIterativeConeAlgorithm(double seed, double radius): 
    theSeedThreshold(seed),
    theConeRadius(radius)
    { }

  /// Find the ProtoJets from the collection of input Candidates.
  void run(const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput);

 private:

  double theSeedThreshold;
  double theConeRadius;
};

#endif
