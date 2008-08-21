import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring( '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-RAW/IDEAL_V5_v3/0001/0408B08E-9162-DD11-B9D5-000423D6B358.root' )
    fileNames = cms.untracked.vstring( '/store/relval/CMSSW_2_1_0/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V4_v1/0000/0EF324BD-9160-DD11-B591-000423D944F8.root' )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

# Conditions: fake or frontier
# process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'STARTUP_V4::All'

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
process.schedule = cms.Schedule( process.DoHltMuon , process.DoHLTJets, process.DoHLTPhoton , process.DoHLTElectron , process.DoHLTElectronStartUpWindows , process.DoHLTElectronLargeWindows , process.DoHLTTau , process.analyzeThis )

