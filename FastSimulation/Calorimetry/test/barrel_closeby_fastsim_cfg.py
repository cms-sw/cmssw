# ---------------------------------------------------------------------------
# Calorimeter-face vertex support for FastSim (CloseByParticleGun).
#
#  Author : Sitian Qian
#  Date   : 21 Aug 2026 (implementation and validation),
#           04 Sep 2026 (pull-request preparation)
#
#  Design inspired by Jan Eysermans' HGCAL FastSim demonstrator
#  (CMSSW_11_3_0_pre3, 2021). Defaults keep the new switches off.
# ---------------------------------------------------------------------------

"""
CloseByParticleGun + FastSim: particles fired FROM the ECAL BARREL face.
Barrel variant of hgcal_closeby_fastsim_cfg.py: same two opt-in switches, vertex at r = 130.

Stock FastSim cannot do this at all: ParticleFilter rejects any primary whose
vertex is outside the tracker volume (r < 129, |z| < 303.353), so a gun vertex
at z = 321 produced zero SimTracks. Two opt-in switches fix it:

  * particleFilter.acceptCaloVertices = True   -- accept calo-region vertices
  * fastSimProducer.caloVertexBackupDistance   -- move the particle back along
    its momentum before the calo-layer navigation, so a vertex sitting exactly
    on (or epsilon past) the HGCAL entrance layer is still picked up by the
    standard machinery and handed to the CalorimetryManager.

The particle never sees the tracker; it goes straight to the calorimetry, which
is Jan's original design for calo-face guns.

Field-off on purpose: the parametrization was derived field-free and a straight
back-shift is exact only without a field.

Note on the gun: this release (20_1) carries the ip==0 guard in
CloseByParticleGunProducer, so the first particle's phi is NOT shifted by
Delta/R -- the 14_0 samples were (phi 1.68299 instead of 1.57).

Run:  cmsRun hgcal_closeby_fastsim_cfg.py [maxEvents=N] [energy=50] [pdgid=22]
"""

import math

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

from Configuration.Eras.Era_Phase2C17I13M9_FastSim_cff import Phase2C17I13M9_FastSim

opts = VarParsing.VarParsing('analysis')
opts.register('energy', 50.0, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.float, 'gun energy [GeV]')
opts.register('pdgid', 22, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, 'particle id: 22 photon, 211 pion')
opts.register('seed', 12345, VarParsing.VarParsing.multiplicity.singleton,
              VarParsing.VarParsing.varType.int, 'RNG seed (must differ per batch job)')
opts.setDefault('maxEvents', 20)
opts.parseArguments()

process = cms.Process('HGCALFS', Phase2C17I13M9_FastSim)

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D110Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T35', '')

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(opts.maxEvents))
process.source = cms.Source('EmptySource')

process.RandomNumberGeneratorService = cms.Service(
    'RandomNumberGeneratorService',
    generator=cms.PSet(initialSeed=cms.untracked.uint32(opts.seed)),
    VtxSmeared=cms.PSet(initialSeed=cms.untracked.uint32(opts.seed + 1)),
    fastSimProducer=cms.PSet(initialSeed=cms.untracked.uint32(opts.seed + 2)),
)

# Vertex ON the ECAL barrel face: just outside the tracker volume
# (r = 129 cm), central rapidity, momentum radially outward (Pointing).
_z0 = 0.0
_r0 = 130.0

process.generator = cms.EDProducer(
    'CloseByParticleGunProducer',
    PGunParameters=cms.PSet(
        PartID=cms.vint32(opts.pdgid),
        # energy window (En, since FlatPtGeneration is off)
        VarMin=cms.double(opts.energy), VarMax=cms.double(opts.energy + 0.001),
        MaxVarSpread=cms.bool(False),
        LogSpacedVar=cms.bool(False),
        FlatPtGeneration=cms.bool(False),
        # position window: controlled by (R, Z), not eta
        ControlledByEta=cms.bool(False),
        ControlledByREta=cms.bool(False),
        RMin=cms.double(_r0 - 0.01), RMax=cms.double(_r0 + 0.01),
        ZMin=cms.double(-0.01), ZMax=cms.double(0.01),
        MinPhi=cms.double(1.57), MaxPhi=cms.double(1.570001),
        Delta=cms.double(10.),          # spacing; irrelevant for one particle
        Pointing=cms.bool(True),        # momentum along the position vector
        Overlapping=cms.bool(False),
        RandomShoot=cms.bool(False),
        NParticles=cms.int32(1),
        UseDeltaT=cms.bool(False),
        TMin=cms.double(0.), TMax=cms.double(0.05),
        OffsetFirst=cms.double(0.),
    ),
    AddAntiParticle=cms.bool(False),
    firstRun=cms.untracked.uint32(1),
    psethack=cms.string('close-by %d E %.0f at EB face' % (opts.pdgid, opts.energy)),
)

process.load('Configuration.StandardSequences.VtxSmearedNoSmear_cff')
process.generatorSmeared = cms.EDProducer('GeneratorSmearedProducer')
process.load('FastSimulation.SimplifiedGeometryPropagator.fastSimProducer_cff')

# The two opt-in switches this test exists for.
process.fastSimProducer.particleFilter.acceptCaloVertices = cms.bool(True)
process.fastSimProducer.caloVertexBackupDistance = cms.double(5.0)  # cm

process.load('RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi')
process.load('Geometry.CaloEventSetup.CaloTopology_cfi')
for _blk in (process.fastSimProducer.trackerDefinition, process.fastSimProducer.caloDefinition):
    if hasattr(_blk, 'trackerAlignmentLabel'):
        _blk.trackerAlignmentLabel = cms.untracked.string('')

process.MessageLogger.cerr.FwkReport.reportEvery = 5
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.CalorimetryManager = cms.untracked.PSet(limit=cms.untracked.int32(-1))

process.out = cms.OutputModule(
    'PoolOutputModule',
    fileName=cms.untracked.string('hgcal_closeby_fastsim.root'),
    outputCommands=cms.untracked.vstring(
        'drop *',
        'keep *_fastSimProducer_*_*',
        'keep *_generatorSmeared_*_*',
        'keep *_generator_*_*',
    ),
)

process.gen = cms.Path(process.generator * process.VtxSmeared * process.generatorSmeared)
process.sim = cms.Path(process.fastSimProducer)
process.outpath = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.gen, process.sim, process.outpath)
