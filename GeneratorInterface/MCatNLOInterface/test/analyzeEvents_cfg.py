import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:mcatnloZee.root')
)

process.myanalysis = cms.EDAnalyzer("ZeeAnalyzer",
                                    OutputFilename = cms.untracked.string('Zee_histos.root'),
                                    hepMCProductTag = cms.InputTag("VtxSmeared"),
                                    genEventInfoProductTag = cms.InputTag("generator")
)

process.p = cms.Path(process.myanalysis)


