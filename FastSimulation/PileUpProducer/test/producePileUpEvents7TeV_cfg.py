import FWCore.ParameterSet.Config as cms
process = cms.Process("PileUp")


process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 123456789

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# for the beamspot
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
)

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    crossSection = cms.untracked.double(71260000000.),
    comEnergy = cms.double(7000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettings = cms.vstring('MSTJ(11)=3     ! Choice of the fragmentation function', 
    'MSTJ(22)=2     ! Decay those unstable particles', 
    'PARJ(71)=10.   ! for which ctau  10 mm', 
    'MSTP(2)=1      ! which order running alphaS', 
    'MSTP(33)=0     ! no K factors in hard cross sections', 
    'MSTP(51)=10042     ! CTEQ6L1 structure function chosen', 
    'MSTP(52)=2     ! work with LHAPDF', 
    'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default', 
    'MSTP(82)=4     ! Defines the multi-parton model', 
    'MSTU(21)=1     ! Check on possible errors during program execution', 
    'PARP(82)=1.8387   ! pt cutoff for multiparton interactions', 
    'PARP(89)=1960. ! sqrts for which PARP82 is set', 
    'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter', 
    'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter', 
    'PARP(90)=0.16  ! Multiple interactions: rescaling power', 
    'PARP(67)=2.5    ! amount of initial-state radiation', 
    'PARP(85)=1.0  ! gluon prod. mechanism in MI', 
    'PARP(86)=1.0  ! gluon prod. mechanism in MI', 
    'PARP(62)=1.25   ! ', 
    'PARP(64)=0.2    ! ', 
    'MSTP(91)=1     !', 
    'PARP(91)=2.1   ! kt distribution', 
    'PARP(93)=15.0  ! '),
# This is a vector of ParameterSet names to be read, in this order
# The first two are in the include files below
# The last one are simply my additional parameters
     parameterSets = cms.vstring('pythiaUESettings',
            'pythiaMinBias',
            'myParameters'),
     pythiaMinBias = cms.vstring('MSEL=0        ! User defined processes',
            'MSUB(11)=1     ! Min bias process', 
            'MSUB(12)=1     ! Min bias process', 
            'MSUB(13)=1     ! Min bias process', 
            'MSUB(28)=1     ! Min bias process', 
            'MSUB(53)=1     ! Min bias process', 
            'MSUB(68)=1     ! Min bias process', 
            'MSUB(92)=1     ! Min bias process, single diffractive', 
            'MSUB(93)=1     ! Min bias process, single diffractive', 
            'MSUB(94)=1     ! Min bias process, double diffractive', 
            'MSUB(95)=1     ! Min bias process'),
        #    Disable the ctau check
        myParameters = cms.vstring('PARJ(71) = -1.')
    )
)

process.GENoutput = cms.OutputModule("PoolOutputModule",
                                     splitLevel = cms.untracked.int32(0),
                                     eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
                                     outputCommands = cms.untracked.vstring(
    'drop *',
    'keep GenRunInfoProduct_generator_*_*',
    'keep GenEventInfoProduct_generator_*_*',
    'keep edmHepMCProduct_generator_*_*',
    'keep edmHepMCProduct_source_*_*', 
    'keep GenFilterInfo_*_*_*',
    'keep *_genParticles_*_*'
    ),
                                     fileName = cms.untracked.string('MinBias7TeV_GEN.root'),
                                     )

process.prodPU = cms.EDProducer("producePileUpEvents",
    PUParticleFilter = cms.PSet(
        # Protons with energy larger than EProton (GeV) are all kept
        EProton = cms.double(5000.0),
        # Particles with |eta| > etaMax (momentum direction at primary vertex) 
        # are not simulated - 7.0 includes CASTOR (which goes to 6.6) 
        etaMax = cms.double(7.0),
        # Charged particles with pT < pTMin (GeV/c) are not simulated
        pTMin = cms.double(0.0),
        # Particles with energy smaller than EMin (GeV) are not simulated
        EMin = cms.double(0.0)
    ),
    PUEventFile = cms.untracked.string('MinBias7TeV.root'),
    SavePileUpEvents = cms.bool(True),
    BunchPileUpEventSize = cms.uint32(1000)
)

#outputType = 'edm'
outputType = 'ntuple'

process.source = cms.Source("EmptySource")
if (outputType=='edm'):
    process.p = cms.Path(process.generator+process.offlineBeamSpot)
    process.GENoutput_step = cms.EndPath(process.GENoutput)
elif (outputType=='ntuple'):
    process.p = cms.Path(process.generator*process.prodPU)
else:
    process.p = cms.Path(process.generator) # just run but save nothing

