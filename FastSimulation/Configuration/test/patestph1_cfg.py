import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# Include DQMStore, needed by the famosSimHits
process.DQMStore = cms.Service( "DQMStore")

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Generate H -> ZZ -> l+l- l'+l'- (l,l'=e or mu), with mH=180GeV/c2
#process.load("Configuration.Generator.H200ZZ4L_cfi")
# Generate ttbar events
process.load("FastSimulation/Configuration/ttbar_cfi")
# Generate multijet events with different ptHAT bins
#  process.load("FastSimulation/Configuration/QCDpt80-120_cfi")
#  process.load("FastSimulation/Configuration/QCDpt600-800_cfi")
# Generate Minimum Bias Events
#  process.load("FastSimulation/Configuration/MinBiasEvents_cfi")
# Generate muons with a flat pT particle gun, and with pT=10.
#process.load("FastSimulation/Configuration/FlatPtMuonGun_cfi")
#process.generator.PGunParameters.MinEta=-1.0
#process.generator.PGunParameters.MaxEta=1.0
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

# this is for phase 1 geometries
process.load('FastSimulation.Configuration.Geometries_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = cms.string('STAR17_61_V1A::All')
# end of phase 1

## this is for phase 2 geometries
#process.load('FastSimulation.Configuration.Geometriesph2_cff')
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
#from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_phase2_BE,noCrossing 
## end phase 2

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
process.validation_step = cms.EndPath(process.tracksValidation)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

process.validation_test = cms.EndPath(process.trackingTruthValid+process.tracksValidation)

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

#process.schedule = cms.Schedule( process.simulation,process.prevalidation_step,process.validation_step,process.endjob_step,process.outpath )
process.schedule = cms.Schedule( process.simulation,process.prevalidation_step,process.validation_step,process.endjob_step)
#process.schedule = cms.Schedule( process.simulation, process.outpath)


# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms

#call to customisation function cust_phase2_BE imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
#process = cust_phase2_BE(process)

#call to customisation function noCrossing imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
#process = noCrossing(process)



# End of customisation functions

