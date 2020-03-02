#flake8: noqa

import sys
from FWCore.ParameterSet.VarParsing import VarParsing

# Setting Input Parameters from Line Command
options = VarParsing ('analysis')
options.register('Mode','Muon',VarParsing.multiplicity.singleton, VarParsing.varType.string,"Option to Run: Muon or Electron")
options.register('Mass',900,VarParsing.multiplicity.singleton, VarParsing.varType.int,"Option to Run: Muon or Electron")
options.register('Era','PreTS2',VarParsing.multiplicity.singleton, VarParsing.varType.string,"Era: by default PreTS2 (PreTS2 and PostTS2)")
options.register('XAngle',120,VarParsing.multiplicity.singleton, VarParsing.varType.int,"XAngle: by default 120 (120, 130, 140, 150 and 160)")
options.parseArguments()

print("")
print("Mode: %s"%options.Mode)
print("Era: %s"%options.Era)
print("Mass: %s"%options.Mass)
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

# minimal logger settings
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# number of events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1000000)
)

# redefine particle generator
process.load("SimCTPPS.Generators.PPXZGenerator_cfi")
process.generator.verbosity = 0
process.generator.m_X = options.Mass
process.generator.m_XZ_min = options.Mass + 100
process.generator.m_X_pr1 = options.Mass - 100
process.generator.decayX = True

if options.Mode == "Muon":
	process.generator.decayZToElectrons = False
	process.generator.decayZToMuons = True
elif options.Mode == "Electron":
	process.generator.decayZToElectrons = True
	process.generator.decayZToMuons = False
else:
        print("#"*100)
        print("Please, try Mode=Muon or Mode=Electron")
        print("#"*100)
        print("")
        sys.exit(0)


# distribution plotter
process.ctppsTrackDistributionPlotter = cms.EDAnalyzer("CTPPSTrackDistributionPlotter",
  tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
  outputFile = cms.string("output.root")
)

# acceptance plotter
process.ctppsAcceptancePlotter = cms.EDAnalyzer("CTPPSAcceptancePlotter",
  tagHepMC = cms.InputTag("generator", "unsmeared"),
  tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),

  rpId_45_F = process.rpIds.rp_45_F,
  rpId_45_N = process.rpIds.rp_45_N,
  rpId_56_N = process.rpIds.rp_56_N,
  rpId_56_F = process.rpIds.rp_56_F,

  outputFile = cms.string("acceptance.root")
)

# generator plots
process.load("SimCTPPS.Generators.PPXZGeneratorValidation_cfi")
process.ppxzGeneratorValidation.tagHepMC = cms.InputTag("generator", "unsmeared")
process.ppxzGeneratorValidation.tagRecoTracks = cms.InputTag("ctppsLocalTrackLiteProducer")
process.ppxzGeneratorValidation.tagRecoProtonsSingleRP = cms.InputTag("ctppsProtons", "singleRP")
process.ppxzGeneratorValidation.tagRecoProtonsMultiRP = cms.InputTag("ctppsProtons", "multiRP")
process.ppxzGeneratorValidation.referenceRPDecId_45 = process.rpIds.rp_45_F
process.ppxzGeneratorValidation.referenceRPDecId_56 = process.rpIds.rp_56_F
process.ppxzGeneratorValidation.outputFile = "ppxzGeneratorValidation.root"

# processing path
process.p = cms.Path(
  process.generator
  * process.beamDivergenceVtxGenerator
  * process.ctppsDirectProtonSimulation

  * process.reco_local
  * process.ctppsProtons

  #* process.ctppsTrackDistributionPlotter
  #* process.ctppsAcceptancePlotter
  #* process.ppxzGeneratorValidation
)


# output configuration
process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("output.root"),
  splitLevel = cms.untracked.int32(0),
  eventAutoFlushCompressedSize=cms.untracked.int32(-900),
  compressionAlgorithm=cms.untracked.string("LZMA"),
  compressionLevel=cms.untracked.int32(9),
  outputCommands = cms.untracked.vstring(
    'drop *',
    'keep edmHepMCProduct_*_*_*',
    'keep CTPPSLocalTrackLites_*_*_*',
    'keep recoForwardProtons_*_*_*'
  )
)

process.outpath = cms.EndPath(process.output)

def UseSettingsZ():
  pass

def UseSettingsGamma():
  process.generator.m_Z_mean = 0
  process.generator.m_Z_gamma = 0
  process.generator.m_XZ_min = options.Mass + 1E-6
  process.generator.m_X_pr1 = options.Mass - 100
  process.generator.p_T_Z_min = 80

UseSettingsZ()
