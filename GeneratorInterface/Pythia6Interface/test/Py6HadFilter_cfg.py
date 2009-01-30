import FWCore.ParameterSet.Config as cms

### from Configuration.GenProduction.PythiaUESettings_cfi import *

process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []
process.MessageLogger.fwkJobReports = []

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5))

process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring('file:ttbar_5flavours_xqcut20_10TeV.lhe')
)

process.generator = cms.EDFilter("Pythia6HadronizerFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(2),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
###       pythiaUESettingsBlock,
        pythiaUESettings = cms.vstring('MSTJ(11)=3     ! Choice of the fragmentation function', 
            'MSTJ(22)=2     ! Decay those unstable particles', 
            'PARJ(71)=10 .  ! for which ctau  10 mm', 
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

        processParameters = cms.vstring('MSEL=0         ! User defined processes', 
                        'PMAS(5,1)=1.5   ! c quark mass',
                        'PMAS(5,1)=4.7   ! b quark mass',
                        'PMAS(6,1)=175.0 ! t quark mass',
                        'MSTP(32)=2      ! Q^2 = sum(m_T^2), iqopt = 1'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    ),
    jetMatching = cms.untracked.PSet(
       scheme = cms.string("Madgraph"),
       mode = cms.string("auto"),
       MEMAIN_etaclmax = cms.double(5.0),
       MEMAIN_qcut = cms.double(30.0),
       MEMAIN_minjets = cms.int32(0),
       MEMAIN_maxjets = cms.int32(3),
       MEMAIN_iexcfile = cms.uint32(0) # only set to 1 if need to perform exclusive matching
    )    
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('TestTTbar.root')
)

process.p = cms.Path(process.generator)
process.p1 = cms.Path(process.randomEngineStateProducer)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.p1, process.outpath)
