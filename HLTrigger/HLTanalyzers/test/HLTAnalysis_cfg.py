import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( 
        '/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/1A0FD639-1B86-DD11-A3C0-000423D99614.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

# Conditions: fake or frontier
# process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_V9::All'

process.load("Configuration.StandardSequences.L1Emulator_cff")
# Choose a menu/prescale/mask from one of the choices
# in L1TriggerConfig.L1GtConfigProducers.Luminosity
#process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")

# Run HLT
#process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.load("HLTrigger.Configuration.HLT_2E30_cff")
#process.schedule = process.HLTSchedule

#process.hltL1gtTrigReport = cms.EDAnalyzer( "L1GtTrigReport",
#    UseL1GlobalTriggerRecord = cms.bool( False ),
#    L1GtRecordInputTag = cms.InputTag( "hltGtDigis" )
#)
#process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
#    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
#)
#process.HLTAnalyzerEndpath = cms.EndPath( process.hltL1gtTrigReport + process.hltTrigReport )
#process.schedule.append(process.HLTAnalyzerEndpath)

# OpenHLT specificss
# Define the HLT reco paths
process.load("HLTrigger.HLTanalyzers.HLTopen_cff")

# Define the analyzer modules
process.load("HLTrigger.HLTanalyzers.HLTAnalyser_cfi")
process.analyzeThis = cms.Path( process.hltanalysis )

# Schedule the whole thing
process.schedule = cms.Schedule( 
    process.DoHltMuon, 
    process.DoHLTJets, 
    process.DoHLTPhoton, 
    process.DoHLTElectron, 
    process.DoHLTElectronStartUpWindows, 
    process.DoHLTElectronLargeWindows, 
    process.DoHLTTau, 
    process.DoHLTBTag,
    process.analyzeThis )

