#flake8: noqa

import sys
from FWCore.ParameterSet.VarParsing import VarParsing

# Setting Input Parameters from Line Command
options = VarParsing ('analysis')
options.register('Era','PreTS2',VarParsing.multiplicity.singleton, VarParsing.varType.string,"Era: by default PreTS2 (PreTS2 and PostTS2)")
options.register('XAngle',120,VarParsing.multiplicity.singleton, VarParsing.varType.int,"XAngle: by default 120 (120, 130, 140, 150 and 160)")
options.parseArguments()

print("")
print("Era: %s"%options.Era)
print("Angle: %s"%options.XAngle)
print("")

import FWCore.ParameterSet.Config as cms

# load config
if options.Era == "PreTS2": 
	from Validation.CTPPS.simu_config.year_2017_preTS2_cff import *
	process = cms.Process('CTPPSDirectSimulation',era)
	process.load("Validation.CTPPS.simu_config.year_2017_preTS2_cff")
elif options.Era == "PostTS2":
	from Validation.CTPPS.simu_config.year_2017_postTS2_cff import *
	process = cms.Process('CTPPSDirectSimulation',era)
	process.load("Validation.CTPPS.simu_config.year_2017_postTS2_cff")
else:
	print("#"*100)
	print("Please, try Era=PreTS2 or Era=PostTS2")
	print("#"*100)
	print("")
	sys.exit(0)

UseCrossingAngle(options.XAngle, process)

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
  fileNames = cms.untracked.vstring("file:miniAOD.root")
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
