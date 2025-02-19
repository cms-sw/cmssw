import FWCore.ParameterSet.Config as cms

process = cms.Process("FamosClusterTest")
# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Famos Common inputs 
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")
process.load("FastSimulation.Configuration.mixNoPU_cfi")
process.GlobalTag.globaltag = cms.string('IDEAL_31X::All') 

# Magnetic Field (new mapping, 3.8 and 4.0T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.famosSimHits.SimulateCalorimetry = False
process.famosSimHits.SimulateTracking = True

process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
process.load("FastSimulation.TrackingRecHitProducer.SiClusterTranslator_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoTracker.Configuration.RecoTracker_cff")

#CPEs
process.load("FastSimulation.TrackingRecHitProducer.FastPixelCPE_cfi")
process.load("FastSimulation.TrackingRecHitProducer.FastStripCPE_cfi")

#First Step
process.load("RecoTracker.IterativeTracking.FirstStep_cff")
process.newClusters.pixelClusters = cms.InputTag('siClusterTranslator')
process.newClusters.stripClusters = cms.InputTag('siClusterTranslator')
process.newPixelRecHits.CPE = cms.string('FastPixelCPE')
process.newStripRecHits.StripCPE = cms.string('FastStripCPE')
process.newMeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.newMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Second Step
process.load("RecoTracker.IterativeTracking.SecStep_cff")
process.secPixelRecHits.CPE = cms.string('FastPixelCPE')
process.secStripRecHits.StripCPE = cms.string('FastStripCPE')
process.secMeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.secMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Third Step
process.load("RecoTracker.IterativeTracking.ThStep_cff")
process.thPixelRecHits.CPE = cms.string('FastPixelCPE')
process.thStripRecHits.StripCPE = cms.string('FastStripCPE')
process.thMeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.thMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Fourth Step
process.load("RecoTracker.IterativeTracking.PixelLessStep_cff")
process.fourthPixelRecHits.CPE = cms.string('FastPixelCPE')
process.fourthStripRecHits.StripCPE = cms.string('FastStripCPE')
process.fourthMeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.fourthMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Fifth Step
process.load("RecoTracker.IterativeTracking.TobTecStep_cff")
process.fifthPixelRecHits.CPE = cms.string('FastPixelCPE')
process.fifthStripRecHits.StripCPE = cms.string('FastStripCPE')
process.fifthMeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.fifthMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Strips
process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi")
process.siStripMatchedRecHits.StripCPE = cms.string('FastStripCPE')
process.siStripMatchedRecHits.ClusterProducer = cms.string('siClusterTranslator')

#Pixels
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")
process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.siPixelRecHits.src = cms.InputTag('siClusterTranslator')
process.siPixelRecHits.CPE = cms.string('FastPixelCPE')
process.load("RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff")
process.globalSeedsFromTriplets.TTRHBuilder = cms.string("FastPixelCPE")

#Transient Rec Hits
process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi")
process.ttrhbwr.StripCPE = cms.string('FastStripCPE')
process.ttrhbwr.PixelCPE = cms.string('FastPixelCPE')
process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi")
process.TTRHBuilderAngleAndTemplate.StripCPE = cms.string('FastStripCPE')
process.TTRHBuilderAngleAndTemplate.PixelCPE = cms.string('FastPixelCPE')
process.load("RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelPairs_cfi")
process.myTTRHBuilderWithoutAngle4PixelPairs.PixelCPE = cms.string("FastPixelCPE")
process.load("RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelTriplets_cfi")
process.myTTRHBuilderWithoutAngle4PixelTriplets.PixelCPE = cms.string("FastPixelCPE")
process.load("RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedPairs_cfi")
process.myTTRHBuilderWithoutAngle4MixedPairs.PixelCPE = cms.string("FastPixelCPE")
process.load("RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedTriplets_cfi")
process.myTTRHBuilderWithoutAngle4MixedTriplets.PixelCPE = cms.string("FastPixelCPE")

#Tracks
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.MeasurementTracker.stripClusterProducer = cms.string('siClusterTranslator')
process.MeasurementTracker.pixelClusterProducer = cms.string('siClusterTranslator')
process.MeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.MeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Making sure not to use the Seed Comparitor
process.newSeedFromTriplets.SeedComparitorPSet.ComponentName = 'none'
process.secTriplets.SeedComparitorPSet.ComponentName = 'none'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.source = cms.Source("PoolSource",
                            debugFlag = cms.untracked.bool(True),
                            debugVebosity = cms.untracked.uint32(10),
                            fileNames = cms.untracked.vstring(
    #10 GeV Muons
    #'/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0008/E60F748A-0558-DE11-99B7-001D09F251E0.root',
    #'/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0008/9E97D7CC-8E57-DE11-A84E-0019B9F70607.root',
    #'/store/relval/CMSSW_3_1_0_pre10/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0008/5E8E28B2-9257-DE11-90DF-001D09F2983F.root'
    
    #10 GeV Pions
    #'/store/relval/CMSSW_3_1_0_pre10/RelValSinglePiPt10/GEN-SIM-RECO/IDEAL_31X_v1/0001/FC4B15EF-505A-DE11-8A8A-003048678AC0.root'

    #1 GeV Pions
    '/store/relval/CMSSW_3_1_0_pre10/RelValSinglePiPt1/GEN-SIM-RECO/IDEAL_31X_v1/0001/4CDF3D15-515A-DE11-8B7A-001A92811744.root'
    )
                            )

process.FirstSecondTrackMerging = cms.EDProducer("QualityFilter",
                                               TrackQuality = cms.string('highPurity'),
                                               recTracks = cms.InputTag("mergeFirstTwoSteps")
                                               )


process.FirstSecondThirdFourthFifthTrackMerging = cms.EDProducer("QualityFilter",
                                                    TrackQuality = cms.string('highPurity'),
                                                    recTracks = cms.InputTag("generalTracks")
                                                    )

process.zeroStepHighPurity = cms.EDProducer("QualityFilter",
                                          TrackQuality = cms.string('highPurity'),
                                          recTracks = cms.InputTag("zeroStepTracksWithQuality")
                                          )
process.firstStepHighPurity = cms.EDProducer("QualityFilter",
                                           TrackQuality = cms.string('highPurity'),
                                           recTracks = cms.InputTag("preMergingFirstStepTracksWithQuality")
                                           )
process.fifthStepHighPurity = cms.EDProducer("QualityFilter",
                                           TrackQuality = cms.string('highPurity'),
                                           recTracks = cms.InputTag("tobtecStep")
                                           )


process.Output = cms.OutputModule("PoolOutputModule",
                                  outputCommands = cms.untracked.vstring('drop *',
                                                                         'keep *_*_*_FamosClusterTest',
                                                                         ),
                                  fileName = cms.untracked.string('FastSim_FullTracking_pi1_3_1_0_pre10.root')
                                  )

process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.destinations = ['detailedInfoFullTk.txt']
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#note: Include process.mix to run on a FastSim RelVal file
#note: Include process.famosWithTrackerHits to run straight off a FullSim file
process.Path = cms.Path(process.famosWithTrackerHits*
                        process.siClusterTranslator*
                        process.siPixelRecHits*
                        process.siStripMatchedRecHits*
                        process.ckftracks*
                        process.zeroStepHighPurity*
                        process.firstStepHighPurity*
                        process.fifthStepHighPurity*
                        process.FirstSecondThirdFourthFifthTrackMerging)
process.o = cms.EndPath(process.Output)
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
#process.FamosRecHitAnalysis.UseCMSSWPixelParametrization = False
