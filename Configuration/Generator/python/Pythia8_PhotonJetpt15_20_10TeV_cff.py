import FWCore.ParameterSet.Config as cms

##from Configuration.Generator.PythiaUESettings_cfi import *

generator = cms.EDFilter("Pythia8GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    crossSection = cms.untracked.double(-1.),
    crossSectionNLO = cms.untracked.double(-1.),
    # doPDGConvert = cms.bool(False), # not sure if the option is valid in Py8
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
    processParameters = cms.vstring('Main:timesAllowErrors    = 10000',
                                    'ParticleDecays:limitTau0 = on',     # Decay those unstable particles
                                    'ParticleDecays:tau0Max   = 10.',    # for which _nominal_ proper lifetime < 10 mm
                                    'PromptPhoton:all         = on',
                                    'PhaseSpace:pTHatMin      = 15.',
                                    'PhaseSpace:pTHatMax      = 20.'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('processParameters')
    )
)

photonfilter = cms.EDFilter("MCSingleParticleFilter",
                          MaxEta = cms.untracked.vdouble(2.4),
                          MinEta = cms.untracked.vdouble(-2.4),
                          MinPt = cms.untracked.vdouble(15.0),
                          ParticleID = cms.untracked.vint32(22)
                          )

ProductionFilterSequence = cms.Sequence(generator*photonfilter)
