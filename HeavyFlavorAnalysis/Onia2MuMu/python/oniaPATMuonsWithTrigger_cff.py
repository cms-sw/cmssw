import FWCore.ParameterSet.Config as cms

#this is our version of the patMuonsWithTrigger from MuonAnalysis, we have rename all methods to avoid any clash, and remove
#all dependencies othen than to PatAlgos.

### ==== Make PAT Muons ====

import PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi
oniaPATMuonsWithoutTrigger = PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi.patMuons.clone(
    muonSource = 'muons',
    # embed the tracks, so we don't have to carry them around
    embedTrack          = True,
    embedCombinedMuon   = True,
    embedStandAloneMuon = True,
    embedPFCandidate    = False,
    embedCaloMETMuonCorrs = cms.bool(False),
    embedTcMETMuonCorrs   = cms.bool(False),
    embedPfEcalEnergy     = cms.bool(False),
    # then switch off some features we don't need
    embedPickyMuon = False,
    embedTpfmsMuon = False, 
    userIsolation = cms.PSet(),   # no extra isolation beyond what's in reco::Muon itself
    isoDeposits = cms.PSet(), # no heavy isodeposits
    addGenMatch = False,       # no mc: T&P doesn't take it from here anyway.
)
# Reset all these; the default in muonProducer_cfi is not empty, but wrong
oniaPATMuonsWithoutTrigger.userData.userInts.src    = []
oniaPATMuonsWithoutTrigger.userData.userFloats.src  = []
oniaPATMuonsWithoutTrigger.userData.userCands.src   = []
oniaPATMuonsWithoutTrigger.userData.userClasses.src = []

### ==== Unpack trigger, and match ====
from PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi import patTrigger as oniaPATTriggerTMP
oniaPATTriggerTMP.onlyStandAlone = True
oniaPATTrigger = cms.EDProducer("TriggerObjectFilterByCollection",
    src = cms.InputTag("oniaPATTriggerTMP"),
    collections = cms.vstring("hltL2MuonCandidates", "hltL3MuonCandidates", "hltHighPtTkMuonCands", "hltGlbTrkMuonCands")
)

### ==== Then perform a match for all HLT triggers of interest
PATmuonTriggerMatchHLT = cms.EDProducer( "PATTriggerMatcherDRDPtLessByR",
    src     = cms.InputTag( "oniaPATMuonsWithoutTrigger" ),
    matched = cms.InputTag( "oniaPATTrigger" ),
    matchedCuts = cms.string(""),
    maxDPtRel = cms.double( 0.5 ),
    maxDeltaR = cms.double( 0.5 ),
    resolveAmbiguities    = cms.bool( True ),
    resolveByMatchQuality = cms.bool( True ) #change with respect to previous tag
)

PATmuonMatchHLTL2   = PATmuonTriggerMatchHLT.clone(matchedCuts = cms.string('coll("hltL2MuonCandidates")'), 
                                                   maxDeltaR = 0.3, maxDPtRel = 10.0)       #maxDeltaR Changed accordingly to Zoltan tuning. It was: 1.2
PATmuonMatchHLTL3   = PATmuonTriggerMatchHLT.clone(matchedCuts = cms.string('coll("hltL3MuonCandidates")'), 
                                                   maxDeltaR = 0.1, maxDPtRel = 10.0)       #maxDeltaR Changed accordingly to Zoltan tuning. It was: 0.5
PATmuonMatchHLTL3T  = PATmuonTriggerMatchHLT.clone(matchedCuts = cms.string('coll("hltGlbTrkMuonCands")'),  
                                                   maxDeltaR = 0.1, maxDPtRel = 10.0)       #maxDeltaR Changed accordingly to Zoltan tuning. It was: 0.5
PATmuonMatchHLTTkMu = PATmuonTriggerMatchHLT.clone(matchedCuts = cms.string('coll("hltHighPtTkMuonCands")'),  
                                                   maxDeltaR = 0.1, maxDPtRel = 10.0)       #maxDeltaR Changed accordingly to Zoltan tuning. It was: 0.5

oniaPATTriggerMatchers1Mu = cms.Sequence(
      PATmuonMatchHLTL2 +
      PATmuonMatchHLTL3 +
      PATmuonMatchHLTL3T +
      PATmuonMatchHLTTkMu
)

oniaPATTriggerMatchers1MuInputTags = [
    cms.InputTag('PATmuonMatchHLTL2'),
    cms.InputTag('PATmuonMatchHLTL3'),
    cms.InputTag('PATmuonMatchHLTL3T'),
    cms.InputTag('PATmuonMatchHLTTkMu'),
]

## ==== Embed ====
oniaPATMuonsWithTrigger = cms.EDProducer( "PATTriggerMatchMuonEmbedder",
    src     = cms.InputTag(  "oniaPATMuonsWithoutTrigger" ),
    matches = cms.VInputTag()
)
oniaPATMuonsWithTrigger.matches += oniaPATTriggerMatchers1MuInputTags

## ==== Trigger Sequence ====
oniaPATTriggerMatching = cms.Sequence(
    oniaPATTriggerTMP * oniaPATTrigger * 
    oniaPATTriggerMatchers1Mu *
    oniaPATMuonsWithTrigger
)

oniaPATMuonsWithTriggerSequence = cms.Sequence(
    oniaPATMuonsWithoutTrigger *
    oniaPATTriggerMatching
)
