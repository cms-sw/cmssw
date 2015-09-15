from RecoBTag.CTagging.pfInclusiveSecondaryVertexFinderCvsLTagInfos_cfi import *
from RecoBTag.CTagging.pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos_cfi import *
from RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff import *
from RecoBTag.CTagging.cTagging_cff import *

# new candidate-based ctagging sequence, requires its own IVF vertices (relaxed IVF reconstruction cuts)
# but IP and soft-lepton taginfos from btagging sequence can be recycled
pfCTagging = cms.Sequence(
    ( inclusiveCandidateVertexingCvsL *
      pfInclusiveSecondaryVertexFinderCvsLTagInfos
    )

    # CSV + soft-lepton variables combined (ctagger optimized for c vs dusg)
    * pfCombinedCvsLJetTags
    * pfCombinedCvsBJetTags
)
