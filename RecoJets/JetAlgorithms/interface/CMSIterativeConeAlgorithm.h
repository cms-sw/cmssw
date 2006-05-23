#ifndef JetAlgorithms_CMSIterativeConeAlgorithm_h
#define JetAlgorithms_CMSIterativeConeAlgorithm_h

/** \class CMSIterativeConeAlgorithm
 *
 * CMSIterativeConeAlgorithm - iterative cone algorithm without 
 * jet merging/splitting. Originally implemented in ORCA by H.P.Wellish.  
 * Documented in CMS NOTE-2006/036
 *
 * \author A.Ulyanov, ITEP
 * $Id$
 ************************************************************/


#include <vector>

#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

class CMSIterativeConeAlgorithm{
 public:
  typedef std::vector <const reco::Candidate*> InputCollection;
  typedef std::vector<ProtoJet> OutputCollection;

  /** Constructor
  \param seed defines the minimum ET in GeV of a tower that can seed a jet.
  \param radius defines the maximum radius of a jet in eta-phi space.
  \param towerThreshold defines the minimum ET in GeV for a tower to be inluded in a jet 
  */
  CMSIterativeConeAlgorithm(double seed, double radius, double towerThreshold): 
    theSeedThreshold(seed),
    theConeRadius(radius),
    theTowerThreshold(towerThreshold)
    { }

  /// Find the ProtoJets from the collection of input Candidates.
  void run(const InputCollection& fInput, OutputCollection* fOutput);

 private:

  double theSeedThreshold;
  double theConeRadius;
  double theTowerThreshold;
};

#endif
