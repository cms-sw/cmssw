# GEN-SIM for the BTL test beam 2025 geometry with active DUT and REF modules (DD4hep),
# adapted from BTLTB2025_step1_DUTbehind_cfg.py. The setup is rotated so that the beam travels along +z (see btl.xml):
# same incidence on the modules and same vertex position relative to them as in the 2025 configuration
#   cmsRun BTLTB2025_step1_cfg.py maxEvents=200 outputFile=BTLTB2025_step1.root
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')
options.maxEvents = 200
options.outputFile = 'BTLTB2025_step1.root'
options.parseArguments()

from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
process = cms.Process('SIM', Phase2C22I13M9, dd4hep)

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Geometry.MTDTBCommonData.GeometryDD4hepExtendedBTLTB2025Reco_DUTbehind_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedBeamProfile_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))
process.source = cms.Source("EmptySource")
process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.uint32(1),
                                     numberOfStreams = cms.untracked.uint32(0))

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('generation_step')),
    dataset = cms.untracked.PSet(dataTier = cms.untracked.string('GEN-SIM'), filterName = cms.untracked.string('')),
    fileName = cms.untracked.string('file:' + options.outputFile),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.genstepfilter.triggerConditions = cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_0T', '')

# Pion beam along +z (eta = 10, i.e. 0.09 mrad from the z axis), starting at z = BeamPosition = 0 with a
# transverse Gaussian profile centred on the z axis. In the rotated geometry the z axis goes through the REF
# and DUT centres (z = 41.2 and 71.2 cm): this is the same beam as in BTLTB2025_step1_DUTbehind_cfg.py
# (phi = 52 deg in the transverse plane, BeamMeanY = -91.015 cm), expressed in the beam frame.
beamEta = 10.
process.VtxSmeared.MinEta = beamEta
process.VtxSmeared.MaxEta = beamEta
process.VtxSmeared.MinPhi = 0.
process.VtxSmeared.MaxPhi = 0.
process.VtxSmeared.BeamMeanX = 0.
process.VtxSmeared.BeamMeanY = 0.
process.VtxSmeared.BeamPosition = 0.
process.VtxSmeared.BeamSigmaX = 2.
process.VtxSmeared.BeamSigmaY = 2.

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinEta = cms.double(beamEta),
        MaxEta = cms.double(beamEta),
        MinPhi = cms.double(0.),
        MaxPhi = cms.double(0.),
        MinE   = cms.double(180.00),
        MaxE   = cms.double(180.00)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(False)
)

# only the MTD sensitive detector is present in the test beam geometry
process.g4SimHits.OnlySDs = ['MtdSensitiveDetector']
# the beam is along z (eta = 10): remove the CMS acceptance cut on the generated particles (abs(eta) < 5.5)
process.g4SimHits.Generator.ApplyEtaCuts = False
# particles of status 1 extrapolated to Z_hector within RDecLenCut of the beam axis are treated as very forward
# (only n and gamma are passed to Geant4): switch it off, the beam is close to the z axis
process.g4SimHits.Generator.RDecLenCut = 0.

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)
process.schedule = cms.Schedule(process.generation_step, process.genfiltersummary_step, process.simulation_step,
                                process.endjob_step, process.FEVTDEBUGoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

for path in process.paths:
    getattr(process, path).insert(0, process.generator)

process.MessageLogger.cerr.FwkReport.reportEvery = 50

from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
