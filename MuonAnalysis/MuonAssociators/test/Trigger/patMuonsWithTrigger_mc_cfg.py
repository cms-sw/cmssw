import FWCore.ParameterSet.Config as cms
process = cms.Process("MuonsWithTrigger")

### standard includes
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000 
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

        
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
#process.load("SimGeneral.TrackingAnalysis.trackingParticlesNoSimHits_cfi")    # On RECO
#process.load("SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi")  # On RECO
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")            # On RAW+RECO
process.load("SimMuon.MCTruth.MuonAssociatorByHitsESProducer_cfi")           # On RAW+RECO

### global tag
process.GlobalTag.globaltag = 'START3X_V26A::All'

### source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'root://pcmssd12.cern.ch//data/gpetrucc/7TeV/hlt/MuHLT_MinBiasMC357_185_1.root',
        'file:/tmp/gpetrucc/MuHLT_MinBiasMC357_185_1.root'
    )
)

### number of events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

### FILTERS for GoodCollision
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed
hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.bscFilter = hltLevel1GTSeed.clone(L1SeedsLogicalExpression = cms.string('(40 OR 41) AND NOT (36 OR 37 OR 38 OR 39)'))
process.bit40     = hltLevel1GTSeed.clone(L1SeedsLogicalExpression = cms.string('(40 OR 41)'))
process.haloVeto  = hltLevel1GTSeed.clone(L1SeedsLogicalExpression = cms.string('NOT (36 OR 37 OR 38 OR 39)'))
process.oneGoodVertexFilter = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof >= 4 && abs(z) <= 15 && position.Rho <= 2"),
   filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)
process.noScraping = cms.EDFilter("FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.25)
)
process.preFilter = cms.Sequence( process.oneGoodVertexFilter * process.noScraping )
process.Skim_GOODCOLL = cms.Path(process.preFilter)
process.Flag_BSC      = cms.Path(process.noScraping * process.oneGoodVertexFilter + process.bscFilter)
process.Flag_Bit40    = cms.Path(process.bit40)
process.Flag_HaloVeto = cms.Path(process.haloVeto)

### Adding Trigger Info from TriggerResultsSummary to the PATMuon
process.load("MuonAnalysis.MuonAssociators.patMuonsWithTrigger_8E29_cff")
from MuonAnalysis.MuonAssociators.patMuonsWithTrigger_8E29_cff import changeTriggerProcessName;
changeTriggerProcessName(process, "HLT2")
#process.muonL1Info.useTrack = 'global'
#process.muonL1Info.useState = 'outermost'
#process.muonMatchHLTL1.useTrack = 'global'
#process.muonMatchHLTL1.useState = 'outermost'

### Adding MCtruth Info to the PATMuon
from MuonAnalysis.MuonAssociators.patMuonsWithTrigger_8E29_cff import addMCinfo
addMCinfo(process)

### Add MC classification by hits
# Requires:
#   SimGeneral/TrackingAnalysis V04-01-00-02 (35X) or V04-01-03+ (37X+)
#   SimTracker/TrackAssociation V01-08-17    (35X+)
#   SimMuon/MCTruth             V02-05-00-01 (35X) or V02-06-00+ (37X+)
process.classByHitsTM = cms.EDProducer("MuonMCClassifier",
    muons = cms.InputTag("muons"),
    trackType = cms.string("segments"),  # or 'inner','outer','global'
    #trackingParticles = cms.InputTag("mergedtruthNoSimHits"),         # RECO Only
    #associatorLabel   = cms.string("muonAssociatorByHits_NoSimHits"), # RECO Only
    trackingParticles = cms.InputTag("mergedtruth"),                 # RAW+RECO
    associatorLabel = cms.string("muonAssociatorByHits"),            # RAW+RECO
)
process.classByHitsGlb = process.classByHitsTM.clone(trackType = "global")

process.classByHits = cms.Sequence(
    process.mix * 
    #process.trackingParticlesNoSimHits *
    process.trackingParticles *
    ( process.classByHitsTM +
      process.classByHitsGlb  )
)
process.patMuonsWithoutTrigger.userData.userInts.src += [
    cms.InputTag("classByHitsTM"),
    cms.InputTag("classByHitsGlb"),
]

### Adding Info about the Muon Station involved to the PATMuon
# Requires MuonAnalysis/Examples V00-03-00+
process.load("MuonAnalysis.Examples.muonStations_cfi")
from MuonAnalysis.Examples.muonStations_cfi import addUserData as addStations
addStations(process.patMuonsWithoutTrigger)


process.p = cms.Path(
    process.preFilter  +
    process.classByHits +
    process.muonStations +
    process.patMuonsWithTriggerSequence 
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('tupleMC.root'),
    outputCommands = cms.untracked.vstring('drop *',
        'keep patMuons_patMuonsWithTrigger__*',                    ## All PAT muons
        'keep edmTriggerResults_TriggerResults_*_*',               ## HLT info, per path (cheap)
        'keep l1extraL1MuonParticles_l1extraParticles_*_*',        ## L1 info (cheap)
        #--- Other things you might want ---
        'keep *_offlinePrimaryVertices__*',                  ## 
        'keep *_offlineBeamSpot__*',                         ##
        'keep recoTrackExtras_standAloneMuons_*_*',          ## track states at the muon system, to which patMuons sta tracks point (useful if you want variables of the innermost or outermost state)
        'keep TrackingRecHitsOwned_standAloneMuons_*_*',     ## muon rechits, to compute things like number of stations
    ),
    SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('Skim_GOODCOLL') )
)
process.e = cms.EndPath(process.out)
