# The following comments couldn't be translated into the new config version:

# This is a vector of ParameterSet names to be read, in this order

# Tau jets (configuration by A. Nikitenko)

import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PythiaSource",
    Phimin = cms.untracked.double(0.0),
    #  possibility to run single or double back-to-back particles with PYTHIA
    # if ParticleID = 0, run PYTHIA
    ParticleID = cms.untracked.int32(-15),
    Etamin = cms.untracked.double(0.0),
    DoubleParticle = cms.untracked.bool(True),
    Phimax = cms.untracked.double(360.0),
    Ptmin = cms.untracked.double(20.0),
    Ptmax = cms.untracked.double(420.0),
    Etamax = cms.untracked.double(2.4),
    pythiaVerbosity = cms.untracked.bool(False),
    PythiaParameters = cms.PSet(
        pythiaUESettings = cms.vstring('MSTJ(11)=3     ! Choice of the fragmentation function', 'MSTJ(22)=2     ! Decay those unstable particles', 'PARJ(71)=10 .  ! for which ctau  10 mm', 'MSTP(2)=1      ! which order running alphaS', 'MSTP(33)=0     ! no K factors in hard cross sections', 'MSTP(51)=7     ! structure function chosen', 'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default', 'MSTP(82)=4     ! Defines the multi-parton model', 'MSTU(21)=1     ! Check on possible errors during program execution', 'PARP(82)=1.9409   ! pt cutoff for multiparton interactions', 'PARP(89)=1960. ! sqrts for which PARP82 is set', 'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter', 'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter', 'PARP(90)=0.16  ! Multiple interactions: rescaling power', 'PARP(67)=2.5    ! amount of initial-state radiation', 'PARP(85)=1.0  ! gluon prod. mechanism in MI', 'PARP(86)=1.0  ! gluon prod. mechanism in MI', 'PARP(62)=1.25   ! ', 'PARP(64)=0.2    ! ', 'MSTP(91)=1     !', 'PARP(91)=2.1   ! kt distribution', 'PARP(93)=15.0  ! '),
        parameterSets = cms.vstring('pythiaUESettings', 'pythiaTauJets'),
        pythiaTauJets = cms.vstring('MDME(89,1)=0      ! no tau->electron', 'MDME(90,1)=0      ! no tau->muon')
    )
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(11),
        mix = cms.untracked.uint32(12345),
        VtxSmeared = cms.untracked.uint32(98765432)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    datasets = cms.untracked.PSet(
        dataset1 = cms.untracked.PSet(
            dataTier = cms.untracked.string('FEVT')
        )
    ),
    fileName = cms.untracked.string('file:tau_jets.root')
)

process.p1 = cms.Path(process.psim)
process.p2 = cms.Path(process.pdigi)
process.p3 = cms.Path(process.reconstruction_plusRS_plus_GSF)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p1,process.p2,process.p3,process.outpath)

process.MessageLogger.cout.threshold = 'ERROR'
process.MessageLogger.cerr.default.limit = 10
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_EMV'

