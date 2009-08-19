import FWCore.ParameterSet.Config as cms

process = cms.Process('SkimHWW')

# Complete Preselection Sequence for 2e2mu analysis

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_31X_V3::All'


# Complete Skim analysis
process.load('HiggsAnalysis/Skimming/higgsToWW2Leptons_SkimPaths_cff')
process.load('HiggsAnalysis/Skimming/higgsToWW2Leptons_OutputModule_cff')
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                            debugFlag = cms.untracked.bool(True),
                            debugVebosity = cms.untracked.uint32(10),
                            fileNames = cms.untracked.vstring('/store/mc/Summer09/TTbar/GEN-SIM-RECO/MC_31X_V3-v1/0024/FE8CFDCF-2F88-DE11-A502-001F29C9D4A8.root')
                           )

# Endpath
process.o = cms.EndPath ( process.higgsToWW2LeptonsOutputModuleRECOSIM)



