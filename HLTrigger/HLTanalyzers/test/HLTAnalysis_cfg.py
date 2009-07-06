import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/E4C24E91-CD57-DE11-90DE-001D09F2532F.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/E469F214-C857-DE11-A950-001D09F252E9.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/DE489DF6-C657-DE11-ABCA-001D09F28D54.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/D4CBC4D7-CC57-DE11-924E-001D09F29619.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/CA9880CE-AC57-DE11-B891-001D09F25208.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/B67BD00F-F257-DE11-95E5-001D09F2B2CF.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/B66C0FD8-CC57-DE11-9593-001D09F297EF.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/B0863555-CA57-DE11-B198-000423D94534.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/98DD9243-B357-DE11-82B4-00304879FBB2.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/906F7F83-C557-DE11-BF54-000423D99AAA.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4EC4DD37-CD57-DE11-8F88-0019B9F705A3.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4ADD5FD0-BB57-DE11-B8F0-001617DBD5AC.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4A2A45E1-AA57-DE11-82AA-0030487A1990.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/404DF239-C357-DE11-849D-001D09F241F0.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/3CB939FA-B757-DE11-BFB5-001D09F24448.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/36647208-C757-DE11-94B1-001D09F24763.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/3447A45D-CD57-DE11-9396-001D09F24489.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/124FFE86-CD57-DE11-92D7-001D09F24259.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/124CD9B7-C457-DE11-87D4-001D09F25393.root',
        '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/081FA023-C457-DE11-BAC8-0019B9F704D6.root'
    )
)

process.maxEvents = cms.untracked.PSet(   input = cms.untracked.int32( 100 )   )

process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# Which AlCa condition for what. Available from pre11
# * DESIGN_31X_V1 - no smearing, alignment and calibration constants = 1.  No bad channels.
# * MC_31X_V1 (was IDEAL_31X) - conditions intended for 31X physics MC production: no smearing,
#   alignment and calibration constants = 1.  Bad channels are masked.
# * STARTUP_31X_V1 (was STARTUP_31X) - conditions needed for HLT 8E29 menu studies: As MC_31X_V1 (including bad channels),
#   but with alignment and calibration constants smeared according to knowledge from CRAFT.
# * CRAFT08_31X_V1 (was CRAFT_31X) - conditions for CRAFT08 reprocessing.
# * CRAFT_31X_V1P, CRAFT_31X_V1H - initial conditions for 2009 cosmic data taking - as CRAFT08_31X_V1 but with different
#   tag names to allow append IOV, and DT cabling map corresponding to 2009 configuration (10 FEDs).
# Meanwhile...:
process.GlobalTag.globaltag = 'MC_31X_V2::All'


process.load('Configuration/StandardSequences/SimL1Emulator_cff')

# OpenHLT specificss
# Define the HLT reco paths
process.load("HLTrigger.HLTanalyzers.HLTopen_cff")
# Remove the PrescaleService which, in 31X, it is expected once HLT_XXX_cff is imported
del process.PrescaleService

# AlCa OpenHLT specific settings

# Define the analyzer modules
process.load("HLTrigger.HLTanalyzers.HLTAnalyser_cfi")
process.analyzeThis = cms.Path( process.hltanalysis )

# Schedule the whole thing
process.schedule = cms.Schedule( 
    process.DoHLTJets, 
    process.DoHltMuon, 
    process.DoHLTPhoton, 
##    process.DoHLTElectron, 
    process.DoHLTElectronStartUpWindows, 
    process.DoHLTElectronLargeWindows, 
##    process.DoHLTTau, 
##    process.DoHLTBTag,
##    process.DoHLTAlCaECALPhiSym,
    process.DoHLTAlCaPi0Eta1E31,
    process.DoHLTIsoTrack,
    process.analyzeThis )
