
### HLT filter
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
FlowCorrHLTFilter = copy.deepcopy(hltHighLevel)
FlowCorrHLTFilter.throw = cms.bool(False)
FlowCorrHLTFilter.HLTPaths = ["HLT_PAPixelTracks_Multiplicity190_*",
                              "HLT_PAPixelTracks_Multiplicity220_*",
                              "HLT_PAPixelTrackMultiplicity100_FullTrack12_*",
                              "HLT_PAPixelTrackMultiplicity130_FullTrack12_*",
                              "HLT_PAPixelTrackMultiplicity160_FullTrack12_*",
                              "HLT_PAHFSumET170_*",
                              "HLT_PAHFSumET210_*"
                              ]


flowCorrCandidateSequence = cms.Sequence( FlowCorrHLTFilter )




