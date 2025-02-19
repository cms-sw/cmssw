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
process.GlobalTag.globaltag = 'START3X_V26A::All'


### source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'rfio:/castor/cern.ch/user/g/gpetrucc/7TeV/DATA/Muons_With_HLTDEBUG.root' 
        #'file:/afs/cern.ch/user/g/gpetrucc/scratch0/mu10/clean/CMSSW_3_5_6/src/hlt.root'
        'file:/tmp/botta/Muons_With_HLTDEBUG.root'
    )
)

### number of events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )



process.load("MuonAnalysis.MuonAssociators.patMuonsWithTrigger_8E29_cff")
from MuonAnalysis.MuonAssociators.patMuonsWithTrigger_8E29_cff import changeTriggerProcessName;
changeTriggerProcessName(process, "HLT2")
#process.muonL1Info.useTrack = 'global'
#process.muonL1Info.useState = 'outermost'
#process.muonMatchHLTL1MuOpen.useTrack = 'global'
#process.muonMatchHLTL1MuOpen.useState = 'outermost'
from MuonAnalysis.MuonAssociators.patMuonsWithTrigger_8E29_cff import addMCinfo
addMCinfo(process)



process.load("MuonAnalysis.MuonAssociators.triggerMatcherToHLTDebug_cfi")
from MuonAnalysis.MuonAssociators.triggerMatcherToHLTDebug_cfi import addUserData
process.matchDebug = process.triggerMatcherToHLTDebug.clone()
addUserData(process.patMuonsWithoutTrigger, "matchDebug")
#process.matchDebug.l1matcherConfig.useTrack = 'global'
#process.matchDebug.l1matcherConfig.useState = 'outermost'

## Skimming: change to fit your requirement
process.muonFilter = cms.EDFilter("PATMuonRefSelector", 
    src = cms.InputTag("patMuonsWithTrigger"), 
    cut = cms.string("isGlobalMuon"), 
    filter = cms.bool(True) 
)

process.p = cms.Path(
    process.matchDebug +
    process.patMuonsWithTriggerSequence +
    process.muonFilter          
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('tupleData.root'),
    outputCommands = cms.untracked.vstring('drop *',
        'keep patMuons_patMuonsWithTrigger__*',                    ## All PAT muons
        'keep edmTriggerResults_TriggerResults_*_*',               ## HLT info, per path (cheap)
        'keep l1extraL1MuonParticles_l1extraParticles_*_*',        ## L1 info (cheap)
        #--- Other things you might want ---
        #'keep *_offlinePrimaryVertices__*',                  ## 
        #'keep *_offlineBeamSpot__*',                         ##
        #'keep recoTrackExtras_standAloneMuons_*_*',          ## track states at the muon system, to which patMuons sta tracks point (useful if you want variables of the innermost or outermost state)
        #'keep TrackingRecHitsOwned_standAloneMuons_*_*',     ## muon rechits, to compute things like number of stations
    ),
    SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p') )
)
process.e = cms.EndPath(process.out)
