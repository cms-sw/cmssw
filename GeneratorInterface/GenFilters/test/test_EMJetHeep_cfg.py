import FWCore.ParameterSet.Config as cms

from Configuration.GenProduction.PythiaUESettings_cfi import *

process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("GeneratorInterface.Pythia6Interface.pythiaDefault_cff")

process.load("Configuration.EventContent.EventContent_cff")


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
        generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

# The following three lines reduce the clutter of repeated printouts
# of the same exception message.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['cerr']
process.MessageLogger.statistics = []
process.MessageLogger.fwkJobReports = []
process.MessageLogger.cerr.threshold = "Warning"


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(3),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=1         ! QCD events',
                    'CKIN(3)= 80.0            ! pt_hat_min',
                    'CKIN(4)=120.0            ! pt_hat_max'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)

process.selection = cms.EDFilter("PythiaFilterEMJetHeep",
#    
      moduleLabel = cms.untracked.string('generator'),
      Minbias     = cms.untracked.bool(False),
      MinEventPt  = cms.untracked.double(80.),
      MaxPhotonEta= cms.untracked.double(2.7),                      
      ConeClust   = cms.untracked.double(0.10),
      ConeIso     = cms.untracked.double(0.50),
      NumPartMin  = cms.untracked.uint32(2),
      dRMin       = cms.untracked.double(0.40),
      MaxEvents   = cms.untracked.int32(1000),
      Debug       = cms.untracked.bool(False)                              
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('EMJetHeep_diem.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN')
    )

)

process.p = cms.Path(process.generator * process.selection)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)
