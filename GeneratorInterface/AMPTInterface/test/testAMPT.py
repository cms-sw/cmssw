import FWCore.ParameterSet.Config as cms

process = cms.Process("ANA")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("GeneratorInterface.HydjetInterface.hydjetDefault_cfi")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1)
                                       )

process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
                                        ignoreTotal=cms.untracked.int32(0),
                                        oncePerEventMode = cms.untracked.bool(False)
                                        )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
'file:amptDefault_cfi_py_GEN.root'
)
                            ) 

process.ana = cms.EDAnalyzer('AMPTAnalyzer'
                             )

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('treefile_ampt.root')
                                   )
process.p = cms.Path(process.ana)




