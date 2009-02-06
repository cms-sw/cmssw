import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        # /RelValTTbar/CMSSW_2_2_0_IDEAL_V9_v1/GEN-SIM-RECO
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v1/0000/28C21C98-18B9-DD11-A601-001617E30D00.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v1/0000/6262739A-17B9-DD11-BEE0-001617E30D00.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v1/0000/82E7934A-B5B9-DD11-A0A1-001617E30CA4.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v1/0000/8655C9E9-14B9-DD11-B733-000423D99264.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v1/0000/AA8D34FF-19B9-DD11-87E8-000423D9A212.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-RECO/IDEAL_V9_v1/0000/C660BDA3-16B9-DD11-8E6B-000423D98800.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
        # /RelValTTbar/CMSSW_2_2_0_IDEAL_V9_v1/GEN-SIM-DIGI-RAW-HLTDEBUG 
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/0428C69F-16B9-DD11-B9F5-000423D98868.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/0E58A24A-14B9-DD11-A323-000423D6CA42.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/1A3BB17F-18B9-DD11-BD3F-000423D99BF2.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/2C14B558-19B9-DD11-B447-000423D996C8.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/2E407C55-18B9-DD11-8F0A-000423D990CC.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/344F0EE2-16B9-DD11-A600-001617DBD224.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/4E0703FB-18B9-DD11-92AC-001617E30F4C.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/56185FE7-16B9-DD11-BF7A-000423D6B2D8.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/5C46B22D-17B9-DD11-9637-000423D98AF0.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/5E1CEC82-19B9-DD11-9353-000423D6C8E6.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/60728BF9-19B9-DD11-98F7-000423D98F98.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/621498C5-15B9-DD11-9AEC-001617C3B6C6.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/62943C76-18B9-DD11-BD84-000423D9970C.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/9C270606-15B9-DD11-AA70-000423D6B42C.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/A4DE1D04-18B9-DD11-A10E-001617DBD472.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/B2F9D18B-17B9-DD11-84D2-001617C3B5E4.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/B623AA6C-17B9-DD11-853D-000423D98B08.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/C62FC61C-1BB9-DD11-921D-000423D98F98.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/C6FE245A-14B9-DD11-B2DB-0016177CA778.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/CA28BC8E-1AB9-DD11-B7D3-000423D98BE8.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/CEB8D2D4-15B9-DD11-801A-000423D998BA.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/DAF174CB-14B9-DD11-A2AE-000423D6CA42.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/E45F5C33-16B9-DD11-BE97-000423D99E46.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/E6F99F65-7CB9-DD11-94E7-000423D996B4.root',
        '/store/relval/CMSSW_2_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/EAC854C4-18B9-DD11-B0A0-000423D94524.root'
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
process.GlobalTag.globaltag = 'STARTUP_30X::All'

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

# AlCa OpenHLT specific settings
process.hltEcalRegionalRestFEDs.Pi0ListToIgnore =  cms.InputTag("hltEcalRegionalPi0FEDs")
process.hltEcalRegionalJetsFEDs.Pi0ListToIgnore =  cms.InputTag("hltEcalRegionalPi0FEDs")
process.hltEcalRegionalEgammaFEDs.Pi0ListToIgnore =  cms.InputTag("hltEcalRegionalPi0FEDs")
process.hltEcalRegionalMuonsFEDs.Pi0ListToIgnore =  cms.InputTag("hltEcalRegionalPi0FEDs")
process.hltEcalRegionalTausFEDs.Pi0ListToIgnore =  cms.InputTag("hltEcalRegionalPi0FEDs")

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
    process.DoHLTAlCaECALPhiSym,
    process.DoHLTAlCaPi0,
    process.analyzeThis )
