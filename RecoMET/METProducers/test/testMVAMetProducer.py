import FWCore.ParameterSet.Config as cms

process = cms.Process("analysis")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10)
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('JetMETCorrections.Configuration.JetCorrectionProducers_cff')
#process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
#process.load('pharris.MVAMet.metProducerSequence_cff')
process.load('RecoMET.METProducers.mvaPFMET_cff')
 
#process.GlobalTag.globaltag = 'GR_R_42_V23::All'
#process.GlobalTag.globaltag = 'MC_44_V12::All'
#process.GlobalTag.globaltag = 'MC_44_V12::All'
process.GlobalTag.globaltag = 'START52_V9::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(300) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/cmst3/user/pharris/HTauTauSynchronization/VBF_HToTauTau_M-120_8TeV-powheg-pythia6-tauola_22612DED-8F8A-E111-92A1-0017A4770034.root'
                            ),
                            skipEvents = cms.untracked.uint32(0)                        
)

process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands = cms.untracked.vstring('keep *'),
                                  fileName = cms.untracked.string("test.root")
)       

process.ana      = cms.Sequence( process.pfMEtMVAsequence)
process.p        = cms.Path(process.ana)
process.outpath  = cms.EndPath(process.output)
