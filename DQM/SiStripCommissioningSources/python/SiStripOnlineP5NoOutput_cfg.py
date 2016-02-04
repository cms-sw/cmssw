import FWCore.ParameterSet.Config as cms

# process declaration
process = cms.Process('SiStripOnline')


#############################################
# General setup
#############################################

# message logger
process.load('DQM.SiStripCommissioningSources.OnlineMessageLogger_cff')

# DQM service
process.load('DQM.SiStripCommissioningSources.OnlineDQM_cff')

# config db settings
process.load('DQM.SiStripCommissioningSources.OnlineSiStripConfigDb_cff')

# input source
process.load('DQM.SiStripCommissioningSources.OnlineSource_cfi')


##############################################
# modules & path for analysis without tracking
##############################################

# tracker digi producer
process.load('EventFilter.SiStripRawToDigi.FedChannelDigis_cfi')

# filter to distinguish between runs not needing or needing tracking
process.load('DQM.SiStripCommissioningSources.TrackingRunTypeFilter_cfi')

# Commissioning source file production
process.load('DQM.SiStripCommissioningSources.CommissioningHistos_cfi')
process.CommissioningHistos.CommissioningTask = 'UNDEFINED'  # <-- run type taken from event data, but can be overriden

# the path to run for analysis without tracking
process.p1 = cms.Path(
    process.FedChannelDigis *
    ~process.trackingRunTypeFilter *
    process.CommissioningHistos
)


#############################################
# setup to prepare tracking
#############################################

# reco inclusion ; cleanup? // was: I'd love to include less, but I fail...
process.load('Configuration.StandardSequences.Reconstruction_cff')
# geometry
process.load('DQM.SiStripCommissioningSources.P5Geometry_cff')
# magnetic field (0T by default)
process.load('MagneticField.Engine.uniformMagneticField_cfi')

# fake global position
process.load('Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff')
# fake conditions for gain
process.load('CalibTracker.Configuration.SiStripGain.SiStripGain_Fake_cff')
# empty quality fake, avoiding RunIfoRcd from DB
process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
process.siStripQualityESProducer.UseEmptyRunInfo = cms.bool(True)
process.siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet()
# fake LA conditions
process.load('CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi')
process.load('CalibTracker.SiStripESProducers.fake.SiStripLorentzAngleFakeESSource_cfi')
# fake conditions for threshold
process.load('CalibTracker.SiStripESProducers.fake.SiStripThresholdFakeESSource_cfi')
process.load('CalibTracker.SiPixelESProducers.SiPixelFakeLorentzAngleESSource_cfi')
# beam spot fake conditions
process.load('RecoVertex.BeamSpotProducer.BeamSpotFakeConditionsNominalCollision_cfi')

# rechit matcher
process.load('RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi')
# CPEs
process.load('RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi')
process.load('RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi')
# TransientTrackingBuilder
process.load('RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi')
process.ttrhbwr.PixelCPE = cms.string('PixelCPEfromTrackAngle')
# MeasurementTracker
process.load('RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cff')
process.MeasurementTracker.StripCPE = cms.string('StripCPEfromTrackAngle')
process.MeasurementTracker.PixelCPE = cms.string('PixelCPEfromTrackAngle')
process.MeasurementTracker.UseStripModuleQualityDB = cms.bool(False)   # read Module status from SiStripQuality
process.MeasurementTracker.UseStripAPVFiberQualityDB = cms.bool(False) # read APV and Fiber status from SiStripQuality
process.MeasurementTracker.UseStripStripQualityDB = cms.bool(False)    # read Strip status from SiStripQuality


#############################################
# modules & path for analysis with tracking
#############################################

# strips digi zero suppression
process.load('RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi')
process.siStripZeroSuppression.RawDigiProducersList = cms.VInputTag(
    cms.InputTag('FedChannelDigis','VirginRaw'), 
    cms.InputTag('FedChannelDigis','ProcessedRaw'),
    cms.InputTag('FedChannelDigis','ScopeMode')
)

# produce clusters from zero suppressed digis
process.load('RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi')
process.siStripClusters.DigiProducersList = cms.VInputTag(
    cms.InputTag('FedChannelDigis','ZeroSuppressed'),
    cms.InputTag('siStripZeroSuppression','VirginRaw'),
    cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
    cms.InputTag('siStripZeroSuppression','ScopeMode')
)

# produce the rechits
process.load('RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi')

# find the seeds
process.load('DQM.SiStripCommissioningSources.P5SeedGenerator_cff')

# reconstruct track candidates
process.load('DQM.SiStripCommissioningSources.P5CosmicCandidateFinder_cff')

# reconstruct tracks
process.load('RecoTracker.TrackProducer.TrackProducer_cfi')
process.TrackProducer.src = cms.InputTag('cosmicCandidateFinder')
process.TrackProducer.Fitter = cms.string('RKFittingSmoother')
process.TrackProducer.TrajectoryInEvent = cms.bool(True)
process.TrackProducer.TTRHBuilder = cms.string('WithTrackAngle')
process.TrackProducer.AlgorithmName = cms.string('cosmic')
process.TrackProducer.alias=('') # can we drop this?

# do the fine delay analysis
process.load('DQM.SiStripCommissioningSources.SiStripFineDelayHit_cfi')

# Commissioning source file production
process.CommissioningHistosWithTracking = process.CommissioningHistos.clone()
process.CommissioningHistosWithTracking.InputModuleLabel = cms.string('siStripFineDelayHit')
process.CommissioningHistosWithTracking.SignalToNoiseCut = cms.double(3.0)

# the path to run for analysis with tracking
process.p2 = cms.Path(
    process.FedChannelDigis *
    process.trackingRunTypeFilter *
    process.offlineBeamSpot *
    process.siStripZeroSuppression *
    process.siStripClusters *
    process.siStripMatchedRecHits *
    process.cosmicseedfinder *
    process.cosmicCandidateFinder *
    process.TrackProducer *
    process.siStripFineDelayHit *
    process.CommissioningHistosWithTracking
)


#############################################
# output
#############################################

#process.load('DQM.SiStripCommissioningSources.OnlineOutput_cfi')
#process.outpath = cms.EndPath( process.consumer )


#############################################

