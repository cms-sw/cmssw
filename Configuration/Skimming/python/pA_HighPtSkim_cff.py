
### HLT filter
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
HighPtHLTFilter = copy.deepcopy(hltHighLevel)
HighPtHLTFilter.throw = cms.bool(False)
HighPtHLTFilter.HLTPaths = ["HLT_PAJet80_NoJetID_v*",
                            "HLT_PAJet100_NoJetID_v*",
                            "HLT_PAJet120_NoJetID_v*",
                            "HLT_PAPhoton40_NoCaloIdVL_v*",
                            "HLT_PAPhoton20_Photon20_NoCaloIdVL_v*",
                            "HLT_PADoubleEle8_CaloIdT_TrkIdVL_v*",
                            "HLT_PAMu12_v*"
                            ]


HighPtCandidateSequence = cms.Sequence( HighPtHLTFilter )




