import FWCore.ParameterSet.Config as cms

process = cms.Process("analysis")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10)
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('JetMETCorrections.Configuration.JetCorrectionProducers_cff')
process.load('RecoMET.METPUSubtraction.mvaPFMET_leptons_cff')

#process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
#process.load('pharris.MVAMet.metProducerSequence_cff')
#process.load('JetMETCorrections.METPUSubtraction.pileupJetIdMVASequence_cff')
 
#process.GlobalTag.globaltag = 'GR_R_42_V23::All'
#process.GlobalTag.globaltag = 'MC_44_V12::All'
#process.GlobalTag.globaltag = 'MC_44_V12::All'
#process.GlobalTag.globaltag = 'START53_V15::All'
process.GlobalTag.globaltag = 'START53_V22::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                            ),
                            skipEvents = cms.untracked.uint32(0)                        
)

process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands = cms.untracked.vstring('keep *'),
                                  fileName = cms.untracked.string("MVaTest.root")
)       

process.ana      = cms.Sequence(process.pfMEtMVAsequence)
process.p        = cms.Path(process.ana)
process.outpath  = cms.EndPath(process.output)

### To add the Jet Id
#+process.pileupJetIdProducer)
