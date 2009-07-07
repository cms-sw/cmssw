import FWCore.ParameterSet.Config as cms

process = cms.Process("FamosClusterTest")
# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Famos Common inputs 
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")
process.load("FastSimulation.Configuration.mixNoPU_cfi")

# Magnetic Field (new mapping, 3.8 and 4.0T)
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
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
process.load("RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cff")
process.globalSeedsFromTripletsWithVertices.TTRHBuilder = cms.string("FastPixelCPE")

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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
                            debugFlag = cms.untracked.bool(True),
                            debugVebosity = cms.untracked.uint32(10),
                            fileNames = cms.untracked.vstring(
    #'/store/relval/CMSSW_3_0_0_pre2/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_V9_v2/0001/4C3E7FD2-1CB4-DD11-BFAB-0016177CA7A0.root'
    #'/store/relval/CMSSW_3_1_0_pre1/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_30X_v1/0001/5CD601D0-5EF4-DD11-BC14-000423D9A212.root'
    '/store/relval/CMSSW_3_1_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/2E564F0D-4BF4-DD11-9983-00304879FBB2.root',
    #'/store/relval/CMSSW_3_1_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/4CF899E9-4AF4-DD11-9CC9-000423D9970C.root',
    #'/store/relval/CMSSW_3_1_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/948D4DFA-49F4-DD11-AD18-0030487D0D3A.root',
    #'/store/relval/CMSSW_3_1_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/E4EEE43D-4AF4-DD11-9E76-001D09F2AF96.root',
    #'/store/relval/CMSSW_3_1_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/E8ED1E7B-4DF4-DD11-9A02-001617C3B6CC.root',
    #'/store/relval/CMSSW_3_1_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/ECEA3F75-5EF4-DD11-AF3B-001617E30D4A.root'
    )
                            )

process.FirstSecondTrackMerging = cms.EDFilter("QualityFilter",
                                               TrackQuality = cms.string('highPurity'),
                                               recTracks = cms.InputTag("mergeFirstTwoSteps")
                                               )


process.FirstSecondThirdFourthTrackMerging = cms.EDFilter("QualityFilter",
                                                    TrackQuality = cms.string('highPurity'),
                                                    recTracks = cms.InputTag("generalTracks")
                                                    )


process.Output = cms.OutputModule("PoolOutputModule",
                                  outputCommands = cms.untracked.vstring('drop *',
                                                                         'keep *_*_*_FamosClusterTest',
                                                                         ),
                                  fileName = cms.untracked.string('FastSim_FullTracking_committry_3_1_0_pre1.root')
                                  )

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.destinations = ['detailedInfoFullTk.txt']
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#note: Include process.mix to run on a FastSim RelVal file
#note: Include process.famosWithTrackerHits to run straight off a FullSim file
process.Path = cms.Path(process.mix*process.siClusterTranslator*process.siPixelRecHits*process.siStripMatchedRecHits*process.iterTracking*process.trackCollectionMerging*process.newCombinedSeeds*process.FirstSecondThirdFourthTrackMerging)
process.o = cms.EndPath(process.Output)
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
#process.FamosRecHitAnalysis.UseCMSSWPixelParametrization = False
