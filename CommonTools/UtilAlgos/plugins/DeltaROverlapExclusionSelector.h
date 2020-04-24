#include "CommonTools/UtilAlgos/interface/MatchByDR.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/OverlapExclusionSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
  edm::View<reco::Candidate> ,
  OverlapExclusionSelector<edm::View<reco::Candidate> ,
                           reco::Candidate, 
                           reco::MatchByDR<reco::Candidate, reco::Candidate> >
  > DeltaROverlapExclusionSelector;

