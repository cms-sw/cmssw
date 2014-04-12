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

### global tag
process.GlobalTag.globaltag = 'GR_R_35X_V7::All'

### source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'rfio:/castor/cern.ch/user/g/gpetrucc/7TeV/DATA/Muons_With_HLTDEBUG.root' 
        #'file:/afs/cern.ch/user/g/gpetrucc/scratch0/mu10/clean/CMSSW_3_5_6/src/hlt.root'
        #'rfio:/castor/cern.ch/user/g/gpetrucc/7TeV/DATA/Muons_With_HLTDEBUG_v9_Run134542_Ls44to52.root'
        #'file:/data/gpetrucc/Feb9Skims/Data_CollisionEvents_MuonSkim.root'
        'root://castorcms.cern.ch//castor/cern.ch/cms/store/data/Commissioning10/MinimumBias/RECO/v9/000/135/149/D6383A41-F85A-DF11-A2A7-0030487C8CBE.root'
    )
)

### number of events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff")
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed
hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.bptxAnd   = hltLevel1GTSeed.clone(L1SeedsLogicalExpression = cms.string('0'))
process.bscFilter = hltLevel1GTSeed.clone(L1SeedsLogicalExpression = cms.string('(40 OR 41) AND NOT (36 OR 37 OR 38 OR 39)'))
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

process.globalMuFilter = cms.EDFilter("TrackCountFilter", src = cms.InputTag("globalMuons"), minNumber = cms.uint32(1))

process.preFilter = cms.Sequence(process.noScraping * process.oneGoodVertexFilter + process.globalMuFilter)
process.Flag_BBPTX = cms.Path(process.noScraping * process.oneGoodVertexFilter + process.bptxAnd)
process.Flag_BSC   = cms.Path(process.noScraping * process.oneGoodVertexFilter + process.bscFilter)

process.load("MuonAnalysis.MuonAssociators.patMuonsWithTrigger_8E29_cff")
#process.muonL1Info.useTrack = 'global'
#process.muonL1Info.useState = 'outermost'
#process.muonMatchHLTL1MuOpen.useTrack = 'global'
#process.muonMatchHLTL1MuOpen.useState = 'outermost'

process.load("MuonAnalysis.Examples.muonStations_cfi");
from MuonAnalysis.Examples.muonStations_cfi import addUserData as addStations
addStations(process.patMuonsWithoutTrigger)

## Skimming: change to fit your requirement
#process.muonFilter = cms.EDFilter("PATMuonRefSelector", 
#    src = cms.InputTag("patMuonsWithTrigger"), 
#    cut = cms.string("isGlobalMuon"), 
#    filter = cms.bool(True) 
#)

process.p = cms.Path(
    process.preFilter  +
    process.muonStations +
    process.patMuonsWithTriggerSequence 
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('tupleData.root'),
    outputCommands = cms.untracked.vstring('drop *',
        'keep patMuons_patMuonsWithTrigger__*',                    ## All PAT muons
        'keep edmTriggerResults_TriggerResults_*_*',               ## HLT info, per path (cheap)
        'keep l1extraL1MuonParticles_l1extraParticles_*_*',        ## L1 info (cheap)
        #--- Other things you might want ---
        'keep *_offlinePrimaryVertices__*',                  ## 
        'keep *_offlineBeamSpot__*',                         ##
        'keep recoTrackExtras_standAloneMuons_*_*',          ## track states at the muon system, to which patMuons sta tracks point (useful if you want variables of the innermost or outermost state)
        #'keep TrackingRecHitsOwned_standAloneMuons_*_*',     ## muon rechits, to compute things like number of stations
    ),
    SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p') )
)
process.e = cms.EndPath(process.out)
