import FWCore.ParameterSet.Config as cms

process = cms.Process("MULTITRACKVALIDATOR")

# message logger
process.MessageLogger = cms.Service("MessageLogger",
     default = cms.untracked.PSet( limit = cms.untracked.int32(10) )
)

# source
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_7_1_0_pre5/RelValSingleMuPt10_UP15/GEN-SIM-RECO/POSTLS171_V1-v1/00000/980D6268-26B6-E311-AA4D-0025905A6136.root' ] );

secFiles.extend( [
       '/store/relval/CMSSW_7_1_0_pre5/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/0AD53A78-F5B5-E311-BF6A-00248C55CC9D.root',
      '/store/relval/CMSSW_7_1_0_pre5/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/1CB3FB16-F6B5-E311-842C-0025905A613C.root',
       '/store/relval/CMSSW_7_1_0_pre5/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/AA4515DD-F5B5-E311-834B-0025905A6076.root',
       '/store/relval/CMSSW_7_1_0_pre5/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/AE114155-F5B5-E311-9837-002618943874.root',
       '/store/relval/CMSSW_7_1_0_pre5/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/F85DDD54-F5B5-E311-98E0-002354EF3BCE.root' ] );

process.source = source
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')
process.GlobalTag.globaltag = 'START71_V1::All'#POSTLS171_V1::All'

### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

## includes for DAF
import RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi
import TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedSeeds_cfi

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.load("RecoTracker.TrackProducer.CTFFinalFitWithMaterialDAF_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
# replace this with whatever track type you want to look at
process.TrackRefitter.TrajectoryInEvent = True
process.RKFittingSmoother.MinNumberOfHits=3
process.TrackRefitter.Fitter = "RKFittingSmoother"
process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")

from RecoTracker.FinalTrackSelectors.TracksWithQuality_cff import *
process.ctfWithMaterialTracksDAF.TrajectoryInEvent = True
process.ctfWithMaterialTracksDAF.src = 'TrackRefitter'
process.ctfWithMaterialTracksDAF.TrajAnnealingSaving = True
process.MRHFittingSmoother.EstimateCut = -1
process.MRHFittingSmoother.MinNumberOfHits = 3

import copy
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
RelaxedChi2 = copy.deepcopy(Chi2MeasurementEstimator)
RelaxedChi2.ComponentName = 'RelaxedChi2'
RelaxedChi2.MaxChi2 = 100.

### validation-specific includes
process.load("SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("DQMServices.Components.DQMFileSaver_cfi")
process.load("Validation.Configuration.postValidation_cff")

process.quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.TrackAssociatorByChi2ESProducer.chi2cut = cms.double(500.0)
process.TrackAssociatorByPullESProducer = process.TrackAssociatorByChi2ESProducer.clone(                      chi2cut = 50.0,
                                onlyDiagonal = True,
                                ComponentName = 'TrackAssociatorByPull')

########### configuration MultiTrackValidator ########
process.dqmSaver.workflow = cms.untracked.string("/SingleMu/DAF/100evs")
#process.multiTrackValidator.outputFile = 'multitrackvalidator_DAF_SingleMuPt10_100evts_AllAssociators.root'
process.multiTrackValidator.associators = ['quickTrackAssociatorByHits','TrackAssociatorByChi2','TrackAssociatorByPull']
process.multiTrackValidator.skipHistoFit=cms.untracked.bool(False)
process.multiTrackValidator.label = ['ctfWithMaterialTracksDAF']
process.multiTrackValidator.UseAssociators = cms.bool(True)
process.multiTrackValidator.runStandalone = cms.bool(True)

process.quickTrackAssociatorByHits.useClusterTPAssociation = cms.bool(True)
process.load("SimTracker.TrackerHitAssociation.clusterTpAssociationProducer_cfi")

### DAF Validator
process.load('SimTracker.TrackAssociation.TrackAssociatorByHits_cfi')
process.load('SimGeneral.TrackingAnalysis.trackingParticles_cfi')

process.demo = cms.EDAnalyzer('DAFValidator',
                tracks = cms.untracked.InputTag('ctfWithMaterialTracksDAF'),
                trackingParticleLabel = cms.InputTag("mix","MergedTrackTruth"),
		associator = cms.untracked.string("TrackAssociatorByChi2"),
		associatePixel = cms.bool(True),
		associateStrip = cms.bool(True),
		associateRecoTracks = cms.bool(True)
)

process.TFileService = cms.Service("TFileService",
               fileName = cms.string('DAFValidator_SingleMuPt10_100evts_AllAssociators.root')
)

process.outputFile = cms.EndPath(process.dqmSaver)

process.validation = cms.Sequence(
#    process.tpClusterProducer *	##it does not compile with DAF::why?
    process.multiTrackValidator 
)

# paths
process.p = cms.Path(
    process.MeasurementTrackerEvent * process.TrackRefitter
    * process.ctfWithMaterialTracksDAF
    * process.validation 
    * process.demo
)

### sequence of paths to run
process.schedule = cms.Schedule(process.p,process.outputFile)


