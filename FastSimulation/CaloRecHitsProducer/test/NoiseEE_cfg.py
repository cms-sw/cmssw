import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD2")

# The number of events to be processed.
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
    
# For valgrind studies
# process.ProfilerService = cms.Service("ProfilerService",
#    lastEvent = cms.untracked.int32(13),
#    firstEvent = cms.untracked.int32(3),
#    paths = cms.untracked.vstring('p1')
#)

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation/Configuration/RandomServiceInitialization_cff")

process.source = cms.Source("PoolSource",
#		            fileNames = cms.untracked.vstring('file:/localscratch/b/beaudett/devel/noise/CMSSW_3_1_0/src/FastSimulation/Configuration/test/MyFirstFamosFile.root')
                            fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_1.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_10.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_11.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_12.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_13.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_14.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_15.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_16.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_17.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_18.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_19.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_2.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_20.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_21.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_22.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_23.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_24.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_25.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_26.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_27.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_28.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_29.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_3.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_30.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_31.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_32.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_33.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_34.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_35.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_36.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_37.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_38.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_39.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_4.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_40.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_41.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_42.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_43.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_44.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_45.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_46.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_47.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_48.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_49.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_5.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_50.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_6.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_7.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_8.root',
                                                              'rfio:/castor/cern.ch/user/b/beaudett/CMSSW310/noise/reco_noise_Full_9.root'),
                             noEventSort = cms.untracked.bool(True),
                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
                            )



# To make histograms
process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''


# Generate H -> ZZ -> l+l- l'+l'- (l,l'=e or mu), with mH=200GeV/c2
#process.load("Configuration.Generator.H200ZZ4L_cfi")
# Generate ttbar events
#  process.load("FastSimulation/Configuration/ttbar_cfi")
# Generate multijet events with different ptHAT bins
#  process.load("FastSimulation/Configuration/QCDpt80-120_cfi")
#  process.load("FastSimulation/Configuration/QCDpt600-800_cfi")
# Generate Minimum Bias Events
#  process.load("FastSimulation/Configuration/MinBiasEvents_cfi")
# Generate muons with a flat pT particle gun, and with pT=10.
#process.load("FastSimulation/Configuration/FlatPtMuonGun_cfi")
#process.generator.PGunParameters.PartID = cms.vint32(12)
# Generate di-electrons with pT=35 GeV
# process.load("FastSimulation/Configuration/DiElectrons_cfi")

# Famos sequences (Frontier conditions)
process.load("FastSimulation/Configuration/CommonInputs_cff")
process.GlobalTag.globaltag = "MC_31X_V1::All"
process.load("FastSimulation/Configuration/FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# If you want to turn on/off pile-up
process.famosPileUp.PileUpSimulator.averageNumber = 0.0    
# You may not want to simulate everything for your study
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
# process.famosSimHits.SimulateMuons = False

process.noiseCheck = cms.EDFilter("NoiseCheck")

# Produce Tracks and Clusters
#process.p1 = cms.Path(process.ProductionFilterSequence*process.famosWithCaloHits*process.noiseCheck)
process.p1 = cms.Path(process.famosWithCaloHits*process.noiseCheck)


# Keep the logging output to a nice level #

#process.Timing =  cms.Service("Timing")
#process.load("FWCore/MessageService/MessageLogger_cfi")
#process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt","cout")
#process.MessageLogger.categories.append("FamosManager")
#process.MessageLogger.cout = cms.untracked.PSet(threshold=cms.untracked.string("INFO"),
#                                                default=cms.untracked.PSet(limit=cms.untracked.int32(0)),
#                                                FamosManager=cms.untracked.PSet(limit=cms.untracked.int32(100000)))

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )
