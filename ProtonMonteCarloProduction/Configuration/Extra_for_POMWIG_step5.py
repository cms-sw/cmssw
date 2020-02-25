import FWCore.ParameterSet.Config as cms

# load common code
from direct_simu_reco_cff import *
process = cms.Process('CTPPSTestAcceptance', era)
process.load("direct_simu_reco_cff")
#SetDefaults(process)
UseCrossingAngle(120, process)
process.load('Configuration.EventContent.EventContent_cff')

# minimal logger settings
#process.MessageLogger = cms.Service("MessageLogger",
  #statistics = cms.untracked.vstring(),
  #destinations = cms.untracked.vstring('cerr'),
  #cerr = cms.untracked.PSet(
  #  threshold = cms.untracked.string('WARNING')
  #)
#)

# event source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring("file:miniAOD_SD.root")
)

# number of events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

# update settings of beam-smearing module
process.beamDivergenceVtxGenerator.src = cms.InputTag("")
process.beamDivergenceVtxGenerator.srcGenParticle = cms.VInputTag(
#    cms.InputTag("genPUProtons","genPUProtons"),
    cms.InputTag("prunedGenParticles")
)

#output file
process.out = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('miniAOD_withProtons.root'),
    outputCommands = process.AODSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.out.outputCommands.extend(
 cms.untracked.vstring(
        'keep *_*_*_*',
        'drop *_ctppsDirectProtonSimulation_*_*',
    )
)

#process.out.outputCommands.append('keep *_*_*_*',
#			          'drop *_ctppsDirectProtonSimulation_*_*')

# processing path
process.p = cms.Path(
  process.beamDivergenceVtxGenerator
  * process.ctppsDirectProtonSimulation
  * process.reco_local
  * process.ctppsProtons
)

process.outpath = cms.EndPath(process.out)

process.schedule=cms.Schedule( process.p, process.outpath)
