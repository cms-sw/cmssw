import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process('PROD',eras.Phase2C4)
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('DataFormats.MuonDetId.gemDetIdAnalyzer_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.GEMAnalysis=dict()

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:step3_29034.root',
#       'root://cms-xrd-global.cern.ch//store/relval/CMSSW_9_1_1_patch1/RelValSingleElectronPt35Extended/GEN-SIM-RECO/91X_upgrade2023_realistic_v1_D17-v1/10000/10D95AC2-B14A-E711-BC4A-0CC47A7C3638.root',
        )
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

# Schedule definition
process.p = cms.Path(process.gemDetIdAnalyzer)
