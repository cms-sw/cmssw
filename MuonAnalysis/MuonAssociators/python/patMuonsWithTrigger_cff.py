import FWCore.ParameterSet.Config as cms

##    __  __       _          ____   _  _____   __  __                       
##   |  \/  | __ _| | _____  |  _ \ / \|_   _| |  \/  |_   _  ___  _ __  ___ 
##   | |\/| |/ _` | |/ / _ \ | |_) / _ \ | |   | |\/| | | | |/ _ \| '_ \/ __|
##   | |  | | (_| |   <  __/ |  __/ ___ \| |   | |  | | |_| | (_) | | | \__ \
##   |_|  |_|\__,_|_|\_\___| |_| /_/   \_\_|   |_|  |_|\__,_|\___/|_| |_|___/
##                                                                           
##   
### ==== Make PAT Muons ====
import PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi
patMuonsWithoutTrigger = PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi.patMuons.clone(
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
    #addTeVRefits = False, ## <<--- this doesn't work. PAT bug ??
    embedPickyMuon = False,
    embedTpfmsMuon = False, 
    userIsolation = cms.PSet(),   # no extra isolation beyond what's in reco::Muon itself
    isoDeposits = cms.PSet(), # no heavy isodeposits
    addGenMatch = False,       # no mc: T&P doesn't take it from here anyway.
)
# Reset all these; the default in muonProducer_cfi is not empty, but wrong
patMuonsWithoutTrigger.userData.userInts.src    = []
patMuonsWithoutTrigger.userData.userFloats.src  = []
patMuonsWithoutTrigger.userData.userCands.src   = []
patMuonsWithoutTrigger.userData.userClasses.src = []

##    __  __       _       _       ____      ___        __  _     _ 
##   |  \/  | __ _| |_ ___| |__   |  _ \    / \ \      / / | |   / |
##   | |\/| |/ _` | __/ __| '_ \  | |_) |  / _ \ \ /\ / /  | |   | |
##   | |  | | (_| | || (__| | | | |  _ <  / ___ \ V  V /   | |___| |
##   |_|  |_|\__,_|\__\___|_| |_| |_| \_\/_/   \_\_/\_/    |_____|_|
##                                                                  
##   
from MuonAnalysis.MuonAssociators.muonL1Match_cfi import muonL1Match as muonL1Info

## Define a generic function, so that it can be used with existing PAT Muons
def addL1UserData(patMuonProducer, l1ModuleLabel = "muonL1Info"):
    "Load variables inside PAT muon, from module <l1ModuleLabel> that you must run before it"
    patMuonProducer.userData.userInts.src += [
        cms.InputTag(l1ModuleLabel, "quality"), # will be -999 in case of no match
    ]
    patMuonProducer.userData.userFloats.src += [  
        cms.InputTag(l1ModuleLabel, "deltaR"),  # will be 999 in case of no match
    ]
    patMuonProducer.userData.userCands.src += [
        cms.InputTag(l1ModuleLabel)
    ]

## Do it for this collection of pat Muons
addL1UserData(patMuonsWithoutTrigger, "muonL1Info")

##    __  __       _       _       _   _ _   _____ 
##   |  \/  | __ _| |_ ___| |__   | | | | | |_   _|
##   | |\/| |/ _` | __/ __| '_ \  | |_| | |   | |  
##   | |  | | (_| | || (__| | | | |  _  | |___| |  
##   |_|  |_|\__,_|\__\___|_| |_| |_| |_|_____|_|  
##                                                 
##   

### ==== Unpack trigger, and match ====
from PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi import patTrigger as patTriggerFull
patTriggerFull.onlyStandAlone = True
patTrigger = cms.EDProducer("TriggerObjectFilterByCollection",
    src = cms.InputTag("patTriggerFull"),
    collections = cms.vstring("hltL1extraParticles", "hltL2MuonCandidates", "hltL3MuonCandidates", "hltGlbTrkMuonCands", "hltMuTrackJpsiCtfTrackCands", "hltMuTrackJpsiEffCtfTrackCands", "hltMuTkMuJpsiTrackerMuonCands"),
) 
#patTrigger = cms.EDFilter("PATTriggerObjectStandAloneSelector",
#    src = cms.InputTag("patTriggerFull"),
#    cut = cms.string('coll("hltL1extraParticles") || coll("hltL2MuonCandidates") || coll("hltL3MuonCandidates") || coll("hltGlbTrkMuonCands") || coll("hltMuTrackJpsiCtfTrackCands") || coll("hltMuTrackJpsiEffCtfTrackCands") || coll("hltMuTkMuJpsiTrackerMuonCands")'),
#) 

### ==== Then perform a match for all HLT triggers of interest
muonTriggerMatchHLT = cms.EDProducer( "PATTriggerMatcherDRDPtLessByR",
    src     = cms.InputTag( "patMuonsWithoutTrigger" ),
    matched = cms.InputTag( "patTrigger" ),
    matchedCuts = cms.string(""),
#    andOr          = cms.bool( False ),
#    filterIdsEnum  = cms.vstring( '*' ),
#    filterIds      = cms.vint32( 0 ),
#    filterLabels   = cms.vstring( '*' ),
#    pathNames      = cms.vstring( '*' ),
#    collectionTags = cms.vstring( '*' ),
    maxDPtRel = cms.double( 0.5 ),
    maxDeltaR = cms.double( 0.5 ),
    resolveAmbiguities    = cms.bool( True ),
    resolveByMatchQuality = cms.bool( True ) #change with respect to previous tag
)

### == For HLT triggers which are just L1s, we need a different matcher
from MuonAnalysis.MuonAssociators.muonHLTL1Match_cfi import muonHLTL1Match
muonMatchL1 = muonHLTL1Match.clone(
    src     = muonTriggerMatchHLT.src,
    matched = muonTriggerMatchHLT.matched,
)

### Single Mu L1
muonMatchHLTL1 = muonMatchL1.clone(matchedCuts = cms.string('coll("hltL1extraParticles")'))
muonMatchHLTL2 = muonTriggerMatchHLT.clone(matchedCuts = cms.string('coll("hltL2MuonCandidates")'), maxDeltaR = 0.3, maxDPtRel = 10.0)  #maxDeltaR Changed accordingly to Zoltan tuning. It was: 1.2
muonMatchHLTL3 = muonTriggerMatchHLT.clone(matchedCuts = cms.string('coll("hltL3MuonCandidates")'), maxDeltaR = 0.1, maxDPtRel = 10.0)  #maxDeltaR Changed accordingly to Zoltan tuning. It was: 0.5
muonMatchHLTL3T = muonTriggerMatchHLT.clone(matchedCuts = cms.string('coll("hltGlbTrkMuonCands")'),  maxDeltaR = 0.1, maxDPtRel = 10.0)  #maxDeltaR Changed accordingly to Zoltan tuning. It was: 0.5
muonMatchHLTCtfTrack  = muonTriggerMatchHLT.clone(matchedCuts = cms.string('coll("hltMuTrackJpsiCtfTrackCands")'),    maxDeltaR = 0.1, maxDPtRel = 10.0)  #maxDeltaR Changed accordingly to Zoltan tuning. 
muonMatchHLTCtfTrack2 = muonTriggerMatchHLT.clone(matchedCuts = cms.string('coll("hltMuTrackJpsiEffCtfTrackCands")'), maxDeltaR = 0.1, maxDPtRel = 10.0)  #maxDeltaR Changed accordingly to Zoltan tuning. 
muonMatchHLTTrackMu  = muonTriggerMatchHLT.clone(matchedCuts = cms.string('coll("hltMuTkMuJpsiTrackerMuonCands")'), maxDeltaR = 0.1, maxDPtRel = 10.0) #maxDeltaR Changed accordingly to Zoltan tuning. 

patTriggerMatchers1Mu = cms.Sequence(
      #muonMatchHLTL1 +   # keep off by default, since it is slow and usually not needed
      muonMatchHLTL2 +
      muonMatchHLTL3 +
      muonMatchHLTL3T 
)
patTriggerMatchers1MuInputTags = [
    #cms.InputTag('muonMatchHLTL1','propagatedReco'), # fake, will match if and only if he muon did propagate to station 2
    #cms.InputTag('muonMatchHLTL1'),
    cms.InputTag('muonMatchHLTL2'),
    cms.InputTag('muonMatchHLTL3'),
    cms.InputTag('muonMatchHLTL3T'),
]

patTriggerMatchers2Mu = cms.Sequence(
    muonMatchHLTCtfTrack  +
    muonMatchHLTCtfTrack2 +
    muonMatchHLTTrackMu
)
patTriggerMatchers2MuInputTags = [
    cms.InputTag('muonMatchHLTCtfTrack'),
    cms.InputTag('muonMatchHLTCtfTrack2'),
    cms.InputTag('muonMatchHLTTrackMu'),
]

## ==== Embed ====
patMuonsWithTrigger = cms.EDProducer( "PATTriggerMatchMuonEmbedder",
    src     = cms.InputTag(  "patMuonsWithoutTrigger" ),
    matches = cms.VInputTag()
)
patMuonsWithTrigger.matches += patTriggerMatchers1MuInputTags
patMuonsWithTrigger.matches += patTriggerMatchers2MuInputTags


## ==== Trigger Sequence ====
patTriggerMatching = cms.Sequence(
    patTriggerFull * patTrigger * 
    patTriggerMatchers1Mu *
    patTriggerMatchers2Mu *
    patMuonsWithTrigger
)

patMuonsWithTriggerSequence = cms.Sequence(
    muonL1Info             *
    patMuonsWithoutTrigger *
    patTriggerMatching
)



def switchOffAmbiguityResolution(process):
    "Switch off ambiguity resolution: allow multiple reco muons to match to the same trigger muon"
    process.muonMatchHLTL1.resolveAmbiguities = False
    process.muonMatchHLTL2.resolveAmbiguities = False
    process.muonMatchHLTL3.resolveAmbiguities = False
    process.muonMatchHLTCtfTrack.resolveAmbiguities = False
    process.muonMatchHLTTrackMu.resolveAmbiguities  = False

def changeTriggerProcessName(process, triggerProcessName, oldProcessName="HLT"):
    "Change the process name under which the trigger was run"
    patTriggerFull.processName = triggerProcessName

def changeRecoMuonInput(process, recoMuonCollectionTag, oldRecoMuonCollectionTag=cms.InputTag("muons")):
    "Use a different input collection of reco muons"
    from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag
    massSearchReplaceAnyInputTag(process.patMuonsWithTriggerSequence, oldRecoMuonCollectionTag, recoMuonCollectionTag)

def useExistingPATMuons(process, newPatMuonTag, addL1Info=False):
    "Start from existing pat Muons instead of producing them"
    process.patMuonsWithTriggerSequence.remove(process.patMuonsWithoutTrigger)
    process.patMuonsWithTriggerSequence.remove(process.muonL1Info)
    process.patMuonsWithTrigger.src = newPatMuonTag
    from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag
    massSearchReplaceAnyInputTag(process.patMuonsWithTriggerSequence, cms.InputTag('patMuonsWithoutTrigger'), newPatMuonTag)
    if addL1Info:
        process.muonL1Info.src = newPatMuonTag.muonSource
        addL1UserData(getattr(process,newPatMuonTag.moduleLabel), 'muonL1Info')

def addPreselection(process, cut):
    "Add a preselection cut to the muons before matching (might be relevant, due to ambiguity resolution in trigger matching!"
    process.patMuonsWithoutTriggerUnfiltered = process.patMuonsWithoutTrigger.clone()
    process.globalReplace('patMuonsWithoutTrigger', cms.EDFilter("PATMuonSelector", src = cms.InputTag('patMuonsWithoutTriggerUnfiltered'), cut = cms.string(cut))) 
    process.patMuonsWithTriggerSequence.replace(process.patMuonsWithoutTrigger, process.patMuonsWithoutTriggerUnfiltered * process.patMuonsWithoutTrigger)

def addMCinfo(process):
    "Add MC matching information to the muons"
    process.load("PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi")
    process.patMuonsWithTriggerSequence.replace(process.patMuonsWithoutTrigger, process.muonMatch + process.patMuonsWithoutTrigger)
    process.patMuonsWithoutTrigger.addGenMatch = True
    process.patMuonsWithoutTrigger.embedGenMatch = True
    process.patMuonsWithoutTrigger.genParticleMatch = 'muonMatch'

def addDiMuonTriggers(process):
    print "[MuonAnalysis.MuonAssociators.patMuonsWithTrigger_cff] Di-muon triggers are already enabled by default"

def addHLTL1Passthrough(process, embedder="patMuonsWithTrigger"):
    process.patMuonsWithTriggerSequence.replace(process.muonMatchHLTL3, process.muonMatchHLTL1 + process.muonMatchHLTL3)
    getattr(process,embedder).matches += [ cms.InputTag('muonMatchHLTL1'), cms.InputTag('muonMatchHLTL1','propagatedReco') ]

def useExtendedL1Match(process, patMuonProd="patMuonsWithoutTrigger", byWhat=["ByQ"]):
    process.load("MuonAnalysis.MuonAssociators.muonL1MultiMatch_cfi")
    process.globalReplace('muonL1Info', process.muonL1MultiMatch.clone(src = process.muonL1Info.src.value()))
    pmp = getattr(process, patMuonProd)
    for X in byWhat:
        pmp.userData.userInts.src   += [ cms.InputTag('muonL1Info', "quality"+X) ]
        pmp.userData.userFloats.src += [ cms.InputTag('muonL1Info', "deltaR"+X) ]
        pmp.userData.userCands.src  += [ cms.InputTag('muonL1Info', X) ]

def useL1MatchingWindowForSinglets(process):
    "Change the L1 trigger matching window to be suitable also for CSC single triggers"
    if hasattr(process, 'muonL1Info'):
        process.muonL1Info.maxDeltaR     = 0.3 #Changed accordingly to Zoltan tuning. It was: 1.2
        process.muonL1Info.maxDeltaEta   = 0.2
        process.muonL1Info.fallbackToME1 = True
    if hasattr(process, 'muonMatchHLTL1'):
        process.muonMatchHLTL1.maxDeltaR     = 0.3 #Changed accordingly to Zoltan tuning. It was: 1.2
        process.muonMatchHLTL1.maxDeltaEta   = 0.2
        process.muonMatchHLTL1.fallbackToME1 = True

