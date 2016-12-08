import FWCore.ParameterSet.Config as cms
from Configuration.Generator.PythiaUEZ2starSettings_cfi import *


generator = cms.EDFilter("Pythia8GeneratorFilter",
   crossSection = cms.untracked.double(1.755e-04),
   maxEventsToPrint = cms.untracked.int32(0),
   pythiaPylistVerbosity = cms.untracked.int32(1),
   filterEfficiency = cms.untracked.double(0.000416),
   pythiaHepMCVerbosity = cms.untracked.bool(False),
   comEnergy = cms.double(13000.0),
   PythiaParameters = cms.PSet(
   processParameters = cms.vstring(

           'Main:timesAllowErrors = 10000',
           'Charmonium:all = on', # turn on charmonium production
           'PartonLevel:MPI = on',
           'SecondHard:generate = on',
           'SecondHard:Charmonium = on',
           #
           'PhaseSpace:pTHatMin = 3.0',
           'PhaseSpace:pTHatMinSecond = 3.0',
           'PhaseSpace:pTHatMinDiverge = 0.5',
           # Modify Singlet decays:
           '445:onMode = off',     # turn off all chi0_c decays
           '445:onIfAny = 443 22', # turn on ksi_c --> Jpsi+gamma
           '10441:onMode = off',   # chi1_c
           '10441:onIfAny = 443 22',
           '20443:onMode = off',
           '20443:onIfAny = 443 22', # chi2_c
           # Modify Octet decays:
           '9940003:onMode = off',
           '9940003:onIfAny = 443 21',
           '9941003:onMode = off',
           '9941003:onIfAny = 443 21',
           '9942003:onMode = off',
           '9942003:onIfAny = 443 21',
           # Allow Jpsi-->2mu only:
           '443:onMode = off',
           '443:onIfAny = -13 13',
           'Tune:pp 5'
       ),
       parameterSets = cms.vstring('processParameters')
   )
)

jpsifilter = cms.EDFilter("MCSingleParticleFilter",
    Status = cms.untracked.vint32(2),
    MinPt = cms.untracked.vdouble(2.0),
    MaxEta = cms.untracked.vdouble(3.0),
    MinEta = cms.untracked.vdouble(-3.0),
    ParticleID = cms.untracked.vint32(443)
)

dimufilter = cms.EDFilter("MCParticlePairFilter",
    Status = cms.untracked.vint32(1, 1),
    MinPt = cms.untracked.vdouble(1.0, 1.0),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    ParticleCharge = cms.untracked.int32(-1),
    ParticleID1 = cms.untracked.vint32(13),
    ParticleID2 = cms.untracked.vint32(13)
)

likemufilter = cms.EDFilter("MCParticlePairFilter",
    Status = cms.untracked.vint32(1, 1),
    MinPt = cms.untracked.vdouble(1.0, 1.0),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    ParticleCharge = cms.untracked.int32(1),
    ParticleID1 = cms.untracked.vint32(13),
    ParticleID2 = cms.untracked.vint32(13)
)


ProductionFilterSequence = cms.Sequence(generator+jpsifilter+dimufilter+likemufilter)

