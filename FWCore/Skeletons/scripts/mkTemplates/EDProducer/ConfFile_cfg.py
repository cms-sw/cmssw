import FWCore.ParameterSet.Config as cms

process = cms.Process("OWNPARTICLES")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:myfile.root'
    )
)

process.myProducerLabel = cms.EDProducer('prodname'
@example_myparticle     , muons = cms.InputTag('muons')
@example_myparticle     , electrons = cms.InputTag('pixelMatchGsfElectrons')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFile.root')
@example_myparticle     , outputCommands = cms.untracked.vstring(
@example_myparticle         'drop *',
@example_myparticle         'keep *_muons_*_*',
@example_myparticle         'keep *_pixelMatchGsfElectrons_*_*', 
@example_myparticle         'keep *_*_*_OWNPARTICLES'
@example_myparticle     )
)

  
process.p = cms.Path(process.myProducerLabel)

process.e = cms.EndPath(process.out)
