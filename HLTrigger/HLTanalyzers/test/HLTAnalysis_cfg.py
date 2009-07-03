import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/EA8E93EE-5C4F-DE11-BE60-000423D6B2D8.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/E4CDBFF2-5C4F-DE11-9144-001D09F2426D.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/E0CD0BBF-524F-DE11-A748-001D09F253D4.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/CC52C9F4-5C4F-DE11-A438-000423D94700.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/94443DEF-5C4F-DE11-870F-001D09F28C1E.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/8C395EF5-5C4F-DE11-8EB1-000423D9939C.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/7A4E997D-6E4F-DE11-A72E-000423D98EC8.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/64B5F5F5-5C4F-DE11-8A33-000423D985E4.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/5C6CCBF3-5C4F-DE11-978B-0030487A1FEC.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/520DEEE9-5C4F-DE11-8DB5-001D09F253D4.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/4492E8EB-5C4F-DE11-ADD8-001D09F2503C.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/3C7C61F5-5C4F-DE11-B7C5-000423D9880C.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/2897F5F0-5C4F-DE11-A77B-001D09F2546F.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValProdTTbar/GEN-SIM-RAW/IDEAL_31X_v1/0007/1A5915F5-5C4F-DE11-8F1F-000423D992DC.root'
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
process.GlobalTag.globaltag = 'IDEAL_31X::All'


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
