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
process.GlobalTag.globaltag = 'POSTLS171_V1::All'

### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

### validation-specific includes
process.load("SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")
process.load("Validation.RecoTrack.cuts_cff")
process.load("Validation.RecoTrack.MultiTrackValidator_cff")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("Validation.Configuration.postValidation_cff")

process.quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.TrackAssociatorByChi2ESProducer.chi2cut = cms.double(500.0)
process.TrackAssociatorByPullESProducer = process.TrackAssociatorByChi2ESProducer.clone(
							chi2cut = 50.0,onlyDiagonal = True,
							ComponentName = 'TrackAssociatorByPull')
 
########### configuration MultiTrackValidator ########
process.multiTrackValidator.outputFile = 'multitrackvalidator_SingleMuPt10_100evts_AssociatorByPull.root'
#process.multiTrackValidator.associators = ['quickTrackAssociatorByHits']
process.multiTrackValidator.associators = ['TrackAssociatorByPull']
process.multiTrackValidator.skipHistoFit=cms.untracked.bool(False)
process.multiTrackValidator.label = ['cutsRecoTracks']
process.multiTrackValidator.UseAssociators = cms.bool(True)
process.multiTrackValidator.runStandalone = cms.bool(True)

process.quickTrackAssociatorByHits.useClusterTPAssociation = cms.bool(True)
process.load("SimTracker.TrackerHitAssociation.clusterTpAssociationProducer_cfi")

process.validation = cms.Sequence(
#    process.tpClusterProducer *  #associate the hits trought an association map between cluster and tp
    process.multiTrackValidator
)

# paths
process.p = cms.Path(
     process.cutsRecoTracks *
     process.validation
)
process.schedule = cms.Schedule(
      process.p
)


