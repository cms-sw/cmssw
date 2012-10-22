import FWCore.ParameterSet.Config as cms

process = cms.Process("FamosRecHitProducer")
# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Famos sequences (Frontier conditions)
process.load("FastSimulation/Configuration/CommonInputs_cff")
process.GlobalTag.globaltag = "MC_31X_V1::All"
process.load("FastSimulation/Configuration/FamosSequences_cff")

# Magnetic Field (new mapping, 3.8 and 4.0T)
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# process.load("FastSimulation.TrackingRecHitProducer.test.FamosRecHitAnalysis_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000)
#    input = cms.untracked.int32(10)
)
process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    firstRun = cms.untracked.uint32(1),
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        # you can request more than 1 particle
        # PartID = cms.vint32(211,11,-13),
        MinE = cms.double(1.0),
        MaxE = cms.double(10.0),
        MinEta = cms.double(-3.0),
        MaxEta = cms.double(3.0),
        MinPhi = cms.double(-3.14159265359), ## it must be in radians
        MaxPhi = cms.double(3.14159265359),
    ),
    AddAntiParticle = cms.bool(False), # back-to-back particles
    Verbosity = cms.untracked.int32(0) ## for printouts, set it to 1 (or greater)   
)

# process.Path = cms.Path(process.famosWithTrackerHits*process.trackerGSRecHitTranslator*process.FamosRecHitAnalysis)
# RecHit Analysis ###
process.load("FastSimulation.TrackingRecHitProducer.FamosRecHitAnalysis_cfi")

process.famosSimHits.SimulateCalorimetry = False
process.famosSimHits.SimulateTracking = True
process.siTrackerGaussianSmearingRecHits.UseCMSSWPixelParametrization = True
process.siTrackerGaussianSmearingRecHits.doRecHitMatching = False

process.p1 = cms.Path(process.generator*
                      process.famosWithTrackerHits*
                      process.trackerGSRecHitTranslator*
                      process.FamosRecHitAnalysis)

process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('rechits.root')
)

process.outpath = cms.EndPath(process.o1)

#process.load("FWCore/MessageService/MessageLogger_cfi")
#process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt","cout")
#process.MessageLogger.categories.append("FamosManager")
#process.MessageLogger.cout = cms.untracked.PSet(threshold=cms.untracked.string("INFO"),
#                                                default=cms.untracked.PSet(limit=cms.untracked.int32(0)),
#                                                FamosManager=cms.untracked.PSet(limit=cms.untracked.int32(100000)))



