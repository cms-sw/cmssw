/** PFCandidateStripMerger

Class that creates strips from Particle Flow Candidates
And outputs a Collection of Candidate Lists
Michail Bachtis
----------------------
University of Wisconsin
bachtis@cern.ch
**/

#include "RecoTauTag/TauTagTools/interface/PFCandidateMergerBase.h"

class PFCandidateStripMerger : public PFCandidateMergerBase
{
 public:
  PFCandidateStripMerger();
  PFCandidateStripMerger(const edm::ParameterSet&);
  ~PFCandidateStripMerger();

  std::vector<reco::PFCandidateRefVector> mergeCandidates(const reco::PFCandidateRefVector&);

 private:

  std::vector<int> inputPdgIds_; //type of candidates to clusterize
  double etaAssociationDistance_;//eta Clustering Association Distance
  double phiAssociationDistance_;//phi Clustering Association Distance
  
  double stripPtThreshold_;

  //Private Methods
  bool candidateMatches(const reco::PFCandidateRef&);
  void sortRefVector(reco::PFCandidateRefVector&);



};
