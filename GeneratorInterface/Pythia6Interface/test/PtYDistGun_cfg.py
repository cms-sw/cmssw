# Particle gun whose pt and rapidity distributions can be specified
# by TGraphs, and later handled by pythia for decays. The example here
# is jpsi, with kinematic distributions from hep-ph/0310274v1
# and chosen to decay into two muons

import FWCore.ParameterSet.Config as cms

### from Configuration.Generator.PythiaUESettings_cfi import *
from GeneratorInterface.Pythia6Interface.pythiaDefault_cff import *

process = cms.Process("Gen")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("Pythia6PtYDistGun",

   maxEventsToPrint = cms.untracked.int32(5),
   pythiaHepMCVerbosity = cms.untracked.bool(True),
   pythiaPylistVerbosity = cms.untracked.int32(1),

   PGunParameters = cms.PSet(                    
      ParticleID = cms.vint32(443),
      kinematicsFile = cms.FileInPath(
         'HeavyIonsAnalysis/Configuration/data/jpsipbpb.root'),                   
      PtBinning = cms.int32(100000),
      YBinning = cms.int32(500),
      MinPt = cms.double(0.0),
      MaxPt = cms.double(100.0),
      MinY = cms.double(-10.0),
      MaxY = cms.double(10.0),
      MinPhi = cms.double(-3.14159265359),
      MaxPhi = cms.double(3.14159265359),
   ),                    

   PythiaParameters = cms.PSet(
      pythiaDefaultBlock,
      jpsiDecay = cms.vstring(
         'BRAT(858) = 0 ! switch off',
         'BRAT(859) = 1 ! switch on',
         'BRAT(860) = 0 ! switch off',
         'MDME(858,1) = 0 ! switch off',
         'MDME(859,1) = 1 ! switch on',
         'MDME(860,1) = 0 ! switch off'),
      upsilonDecay = cms.vstring(
         'BRAT(1034) = 0 ! switch off',
         'BRAT(1035) = 1 ! switch on',
         'BRAT(1036) = 0 ! switch off',
         'BRAT(1037) = 0 ! switch off',
         'BRAT(1038) = 0 ! switch off',
         'BRAT(1039) = 0 ! switch off',
         'BRAT(1040) = 0 ! switch off',
         'BRAT(1041) = 0 ! switch off',
         'BRAT(1042) = 0 ! switch off',
         'MDME(1034,1) = 0 ! switch off',
         'MDME(1035,1) = 1 ! switch on',
         'MDME(1036,1) = 0 ! switch off',
         'MDME(1037,1) = 0 ! switch off',
         'MDME(1038,1) = 0 ! switch off',
         'MDME(1039,1) = 0 ! switch off',
         'MDME(1040,1) = 0 ! switch off',
         'MDME(1041,1) = 0 ! switch off',
         'MDME(1042,1) = 0 ! switch off'),
      parameterSets = cms.vstring('pythiaDefault','jpsiDecay')
   )

)

# For upsilon generation, add in your configuration the uncommented lines :
#
#process.generator.PGunParameters.kinematicsFile = cms.FileInPath('HeavyIonsAnalysis/Configuration/data/upsipbpb.root')
#process.generator.PGunParameters.ParticleID = cms.vint32(553)

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('gen_ptydist_jpsi.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.FEVT)
### process.schedule = cms.Schedule(process.p,process.outpath)
process.schedule = cms.Schedule(process.p)

