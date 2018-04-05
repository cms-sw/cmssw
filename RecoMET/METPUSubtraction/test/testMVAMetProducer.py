import FWCore.ParameterSet.Config as cms

process = cms.Process("analysis")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10)
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load('JetMETCorrections.Configuration.JetCorrectionProducers_cff')
process.load('RecoMET.METPUSubtraction.mvaPFMET_cff')

# Other statements
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
"root://eoscms//eos/cms/store/relval/CMSSW_7_5_0_pre4/RelValZMM_13/GEN-SIM-RECO/PU50ns_MCRUN2_75_V0-v1/00000/24B4BCD7-22F8-E411-A740-0025905A48F2.root"
        ),
                            skipEvents = cms.untracked.uint32(0)                        
)

process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands = cms.untracked.vstring('keep *'),
                                  fileName = cms.untracked.string("MVaTest.root")
)       

process.ana      = cms.Sequence(process.pfMVAMEtSequence)
process.p        = cms.Path(process.ana)
process.outpath  = cms.EndPath(process.output)
