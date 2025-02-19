
### HLT filter
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
DileptoHLTFilter = copy.deepcopy(hltHighLevel)
DileptoHLTFilter.throw = cms.bool(False)
DileptoHLTFilter.HLTPaths = ["HLT_PAL1DoubleMuOpen_v*","HLT_PAL1DoubleMu0_HighQ_v*","HLT_PAL2DoubleMu3_v*","HLT_PAMu3_v*","HLT_PAMu7_v*","HLT_PAMu12_v*"]

## Good Muons
goodMuons = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("isTrackerMuon && track.hitPattern.trackerLayersWithMeasurement > 5 && innerTrack.hitPattern.pixelLayersWithMeasurement > 1 && innerTrack.normalizedChi2 < 1.8"),
)

## psi candidates
psiCandidates = cms.EDProducer("CandViewShallowCloneCombiner",
        decay = cms.string("goodMuons@+ goodMuons@-"),
        cut = cms.string("2.5 < mass < 4.5")
)

psiFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag('psiCandidates'),
    minNumber = cms.uint32(1),
)

## Y candidates
upsCandidates = cms.EDProducer("CandViewShallowCloneCombiner",
        decay = cms.string("goodMuons@+ goodMuons@-"),
        cut = cms.string("7.0 < mass < 14.0")
)

upsFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag('upsCandidates'),
    minNumber = cms.uint32(1),
)

# Z candidates
ZCandidates = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("goodMuons@+ goodMuons@-"),
    cut = cms.string("60.0 < mass < 120.0")
)

ZFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag('ZCandidates'),
    minNumber = cms.uint32(1),
)

psiCandidateSequence = cms.Sequence( DileptoHLTFilter * goodMuons * psiCandidates * psiFilter)

upsCandidateSequence = cms.Sequence( DileptoHLTFilter * goodMuons * upsCandidates * upsFilter)

ZCandidateSequence = cms.Sequence( DileptoHLTFilter * goodMuons * ZCandidates * ZFilter)





