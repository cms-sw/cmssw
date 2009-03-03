# The following comments couldn't be translated into the new config version:

# Force J/Psi-> ee & mumu decay 

import FWCore.ParameterSet.Config as cms

process = cms.Process("Gen")
# this example configuration offers some minimum
# annotation, to help users get through; please
# don't hesitate to read through the comments
# use MessageLogger to redirect/suppress multiple
# service messages coming from the system
#
# in this config below, we use the replace option to make
# the logger let out messages of severity ERROR (INFO level
# will be suppressed), and we want to limit the number to 10
#
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.Generator.PythiaUESettings_cfi")

# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("Pythia6EGun",
    maxEventsToPrint = cms.untracked.int32(5),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    PGunParameters = cms.PSet(
       ParticleID = cms.vint32(443,443),
       AddAntiParticle = cms.bool(False),
       MinPhi = cms.double(-3.14159265359),
       MaxPhi = cms.double(3.14159265359),
       MinE = cms.double(0.0),
       MaxE = cms.double(50.0),
       MinEta = cms.double(0.0),
       MaxEta = cms.double(2.4)
    ),
    PythiaParameters = cms.PSet(
        process.pythiaUESettingsBlock,
        pythiaJpsiDecays = cms.vstring('MDME(858,1)=1                 ! J/psi -> ee turned ON', 
            'MDME(859,1)=1                 ! J/psi -> mumu turned ON', 
            'MDME(860,1)=0                 ! J/psi -> random turned OFF'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'pythiaJpsiDecays')
    )
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('gen_jpsi.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p,process.outpath)


