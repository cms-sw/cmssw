import FWCore.ParameterSet.Config as cms

#GEOM="phase1"
#GEOM="phase2BE"
#GEOM="phase1forward"
#GEOM="phase2BEforward"
GEOM="phase2TkBE5DPixel10D"

process = cms.Process("PROD")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)

# Include DQMStore, needed by the famosSimHits
process.DQMStore = cms.Service( "DQMStore")

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Generate H -> ZZ -> l+l- l'+l'- (l,l'=e or mu), with mH=180GeV/c2
#process.load("Configuration.Generator.H200ZZ4L_cfi")
# Generate ttbar events
#process.load("FastSimulation/Configuration/ttbar_cfi")
# Generate multijet events with different ptHAT bins
#  process.load("FastSimulation/Configuration/QCDpt80-120_cfi")
#  process.load("FastSimulation/Configuration/QCDpt600-800_cfi")
# Generate Minimum Bias Events
#  process.load("FastSimulation/Configuration/MinBiasEvents_cfi")
# Generate muons with a flat pT particle gun, and with pT=10.
process.load("FastSimulation/Configuration/FlatPtMuonGun_cfi")
#process.generator.PGunParameters.MinPt=2.0
#process.generator.PGunParameters.MaxPt=2.0
process.generator.PGunParameters.MinEta=-4.5
process.generator.PGunParameters.MaxEta=4.5
# process.generator.PGunParameters.MinPhi=1.
# process.generator.PGunParameters.MaxPhi=1.

#replace FlatRandomPtGunSource.PGunParameters.PartID={130}

# Generate di-electrons with pT=35 GeV
#process.load("FastSimulation/Configuration/DiElectrons_cfi")

# if you need timing
#process.Timing = cms.Service("Timing")
#process.options = cms.untracked.PSet(
#    wantSummary = cms.untracked.bool(True)
#    )
# Famos sequences (MC conditions, not Fake anymore!)
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")

# needed for DQM
process.load('FastSimulation.Configuration.EventContent_cff')
process.load('FastSimulation.Configuration.Validation_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# If you want to turn on/off pile-up
#process.load('FastSimulation.PileUpProducer.PileUpSimulator_2012_Startup_inTimeOnly_cff')
process.load("FastSimulation.PileUpProducer.PileUpSimulator_NoPileUp_cff")
#process.load('FastSimulation.PileUpProducer.mix_2012_Startup_inTimeOnly_cff')
#process.famosPileUp.PileUpSimulator.averageNumber = 5.0
# You may not want to simulate everything for your study
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True

# this is for phase 1 geometries
if GEOM=="phase1":
    process.load('FastSimulation.Configuration.Geometries_cff')
    from Configuration.AlCa.autoCond import autoCond
    process.GlobalTag.globaltag = cms.string('DES17_62_V7::All')
elif GEOM=="phase2BE":

## this is for phase 2 geometries
    process.load('FastSimulation.Configuration.Geometriesph2_cff')
    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
    from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_phase2_BE,noCrossing 
##turning off material effects (needed ONLY for phase2, waiting for tuning)
    process.famosSimHits.MaterialEffects.PairProduction = cms.bool(False)
    process.famosSimHits.MaterialEffects.Bremsstrahlung = cms.bool(False)
    process.famosSimHits.MaterialEffects.MuonBremsstrahlung = cms.bool(False)
    process.famosSimHits.MaterialEffects.EnergyLoss = cms.bool(False)
    process.famosSimHits.MaterialEffects.MultipleScattering = cms.bool(False)
# keep NI so to allow thickness to be properly treated in the interaction geometry
    process.famosSimHits.MaterialEffects.NuclearInteraction = cms.bool(True)
    process.KFFittingSmootherWithOutlierRejection.EstimateCut = cms.double(50.0)
elif GEOM=="phase1forward":
## this is for phase 2 geometries
    process.load('FastSimulation.Configuration.Geometriesph1Forward_cff')
    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
    from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_phase2_BE,noCrossing 
##turning off material effects (needed ONLY for phase2, waiting for tuning)
    process.famosSimHits.MaterialEffects.PairProduction = cms.bool(False)
    process.famosSimHits.MaterialEffects.Bremsstrahlung = cms.bool(False)
    process.famosSimHits.MaterialEffects.MuonBremsstrahlung = cms.bool(False)
    process.famosSimHits.MaterialEffects.EnergyLoss = cms.bool(False)
    process.famosSimHits.MaterialEffects.MultipleScattering = cms.bool(False)
# keep NI so to allow thickness to be properly treated in the interaction geometry
    process.famosSimHits.MaterialEffects.NuclearInteraction = cms.bool(True)
    process.KFFittingSmootherWithOutlierRejection.EstimateCut = cms.double(50.0)
elif GEOM=="phase2BEforward":
## this is for phase 2 geometries
    process.load('FastSimulation.Configuration.Geometriesph2Forward_cff')
    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
    from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_phase2_BE,noCrossing 
##turning off material effects (needed ONLY for phase2, waiting for tuning)
    process.famosSimHits.MaterialEffects.PairProduction = cms.bool(False)
    process.famosSimHits.MaterialEffects.Bremsstrahlung = cms.bool(False)
    process.famosSimHits.MaterialEffects.MuonBremsstrahlung = cms.bool(False)
    process.famosSimHits.MaterialEffects.EnergyLoss = cms.bool(False)
    process.famosSimHits.MaterialEffects.MultipleScattering = cms.bool(False)
# keep NI so to allow thickness to be properly treated in the interaction geometry
    process.famosSimHits.MaterialEffects.NuclearInteraction = cms.bool(True)
    process.KFFittingSmootherWithOutlierRejection.EstimateCut = cms.double(50.0)
elif GEOM=="phase2TkBE5DPixel10D":
    process.load('FastSimulation.Configuration.GeometriesPhase2TkBE5DPixel10D_cff')
    from Configuration.AlCa.autoCond import autoCond
    process.GlobalTag.globaltag = cms.string('DES17_62_V7::All')
    process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_BarrelEndcap5DPixel10D_cff')
    process.trackerNumberingSLHCGeometry.layerNumberPXB = cms.uint32(20)
    process.trackerTopologyConstants.pxb_layerStartBit = cms.uint32(20)
    process.trackerTopologyConstants.pxb_ladderStartBit = cms.uint32(12)
    process.trackerTopologyConstants.pxb_moduleStartBit = cms.uint32(2)
    process.trackerTopologyConstants.pxb_layerMask = cms.uint32(15)
    process.trackerTopologyConstants.pxb_ladderMask = cms.uint32(255)
    process.trackerTopologyConstants.pxb_moduleMask = cms.uint32(1023)
    process.trackerTopologyConstants.pxf_diskStartBit = cms.uint32(18)
    process.trackerTopologyConstants.pxf_bladeStartBit = cms.uint32(12)
    process.trackerTopologyConstants.pxf_panelStartBit = cms.uint32(10)
    process.trackerTopologyConstants.pxf_moduleMask = cms.uint32(255)
else:
    print "GEOM is undefined or ill-defined, stopping here"
    sys.exit(1)



#process.load('FastSimulation.Configuration.HLT_GRun_cff')

process.load('RecoParticleFlow.PFTracking.pfTrack_cfi')
process.pfTrack.TrajInEvents = cms.bool(True)
process.load('RecoParticleFlow.PFProducer.particleFlowSimParticle_cff')

#Rechit validation
process.load("FastSimulation.TrackingRecHitProducer.GSRecHitValidation_cfi")
process.testanalyzer.outfilename = cms.string('RecHitValidation.root') 

# Famos with everything !
#process.p1 = cms.Path(process.ProductionFilterSequence*process.famosWithEverything)
process.source = cms.Source("EmptySource")
#process.simulation = cms.Path(process.generator*process.famosWithEverything)
#process.simulation = cms.Path(process.generator*process.famosWithTrackerHits)

#process.simulation = cms.Path(process.generator*process.famosWithTracks)

#print process.famosWithEverything
#process.famosWithEverything.remove(process.famosMuonSequence)
#process.famosWithEverything.remove(process.famosMuonIdAndIsolationSequence)
#process.famosWithEverything.remove(process.muonshighlevelreco)
#print process.famosWithEverything

#process.simulation = cms.Path(process.generator*process.famosWithEverything*process.particleFlowSimParticle)
#print process.csc2DRecHits.readBadChambers
process.csc2DRecHits.readBadChannels = cms.bool(False)
#process.csc2DRecHits.readBadChambers = cms.bool(False)
process.simulation = cms.Path(process.generator*process.famosWithEverything)

#process.simulation = cms.Path(process.generator*process.famosWithEverything)

# this is the one that is working
#process.simulation = cms.Path(process.generator*process.famosWithTracks*process.testanalyzer)


# To write out events (not need: FastSimulation _is_ fast!)
#process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *', 
                                           'drop *_mix_*_*'),
#                              process.AODSIMEventContent,
                              fileName = cms.untracked.string('MyFirstFamosFile_2.root')
)

process.load("RecoParticleFlow.Configuration.Display_EventContent_cff")
process.display = cms.OutputModule("PoolOutputModule",
    process.DisplayEventContent,
    fileName = cms.untracked.string('display.root')

)

process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('DQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQM')
    )
)

process.prevalidation_step = cms.Path(process.prevalidation)
process.validation_step = cms.EndPath(process.tracksValidationFS)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

process.validation_test = cms.EndPath(process.trackingTruthValid+process.tracksValidationFS)

process.trackValidator.outputFile='trackvalidation.root'
process.trackValidator.associators = cms.vstring('TrackAssociatorByChi2','TrackAssociatorByHitsRecoDenom')

#process.outpath = cms.EndPath(process.o1*process.display*process.DQMoutput)
# If we keep the trackvalidation.root file we don't need the dqm output
#process.outpath = cms.EndPath(process.o1*process.display)
#process.outpath = cms.EndPath(process.display)
process.outpath = cms.EndPath(process.o1)

# Keep output to a nice level
# process.Timing =  cms.Service("Timing")
# process.load("FWCore/MessageService/MessageLogger_cfi")
# process.MessageLogger.destinations = cms.untracked.vstring("detailedInfo.txt")

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

process.schedule = cms.Schedule( process.simulation,process.prevalidation_step,process.validation_step,process.endjob_step)
