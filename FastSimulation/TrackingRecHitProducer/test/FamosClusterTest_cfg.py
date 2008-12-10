import FWCore.ParameterSet.Config as cms

process = cms.Process("FamosClusterTest")
# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Famos Common inputs 
process.load("FastSimulation.Configuration.CommonInputsFake_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")

# Magnetic Field (new mapping, 3.8 and 4.0T)
# include "Configuration/StandardSequences/data/MagneticField_38T.cff"
process.load("Configuration.StandardSequences.MagneticField_40T_cff")

process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
process.load("FastSimulation.TrackingRecHitProducer.SiClusterTranslator_cfi")
process.load("Configuration.StandardSequences.FakeConditions_cff")
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
#process.secClusters.pixelClusters = cms.InputTag('siClusterTranslator')
#process.secClusters.stripClusters = cms.InputTag('siClusterTranslator')


#Third Step
process.load("RecoTracker.IterativeTracking.ThStep_cff")
process.thPixelRecHits.CPE = cms.string('FastPixelCPE')
process.thStripRecHits.StripCPE = cms.string('FastStripCPE')
process.thMeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.thMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Fourth Step
process.load("RecoTracker.IterativeTracking.PixelLessStep_cff")
process.fourthMeasurementTracker.StripCPE = cms.string('FastStripCPE')
process.fourthMeasurementTracker.PixelCPE = cms.string('FastPixelCPE')

#Strips
process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi")
process.siStripMatchedRecHits.StripCPE = cms.string('FastStripCPE')
process.siStripMatchedRecHits.ClusterProducer = cms.string('siClusterTranslator')

#Pixels
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")
process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.siPixelRecHits.src = cms.InputTag('siClusterTranslator')
process.siPixelRecHits.CPE = cms.string('FastPixelCPE')
process.load("RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi")
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
    input = cms.untracked.int32(-1)
)

### standard includes
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

### validation-specific includes
#process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("Validation.RecoTrack.cuts_cff")
#process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.source = cms.Source("PoolSource",
                            debugFlag = cms.untracked.bool(True),
                            debugVebosity = cms.untracked.uint32(10),
                            fileNames = cms.untracked.vstring(
    'Your FastSim Input File'
    )
                            )

process.FirstSecondTrackMerging = cms.EDFilter("QualityFilter",
                                               TrackQuality = cms.string('highPurity'),
                                               recTracks = cms.InputTag("mergeFirstTwoSteps")
                                               )


process.FirstSecondThirdTrackMerging = cms.EDFilter("QualityFilter",
                                                    TrackQuality = cms.string('highPurity'),
                                                    recTracks = cms.InputTag("generalTracks")
                                                    )


process.Output = cms.OutputModule("PoolOutputModule",
                                  outputCommands = cms.untracked.vstring('drop *',
                                                                         'keep *_*_*_FamosClusterTest',
                                                                         ),
                                  fileName = cms.untracked.string('Your Output File')
                                  )

process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.destinations = ['detailedInfoFullTk.txt']
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.Path = cms.Path(process.siClusterTranslator*process.siPixelRecHits*process.siStripMatchedRecHits*process.ckftracks*process.FirstSecondThirdTrackMerging)
process.o = cms.EndPath(process.Output)
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
#process.FamosRecHitAnalysis.UseCMSSWPixelParametrization = False
