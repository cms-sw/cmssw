import FWCore.ParameterSet.Config as cms

##################################################################
# Put here the globaltag the file name and the number of events:

gtag=cms.string('IDEAL_30X::All')

inputfiles=cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre6/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0002/B2B57D9D-1933-DE11-A285-000423D99CEE.root')
#'file:/d1/dorbaker/test_mu_10GeV_3_1_0_pre1.root')
secinputfiles=cms.untracked.vstring()
nevents=cms.untracked.int32(10)
###################################################################

process = cms.Process("TrackerValidationOnly")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['detailedInfoRecHitVal.txt']
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = gtag
process.load("SimTracker.Configuration.SimTracker_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.load("Validation.TrackerRecHits.trackerRecHitsValidation_cff")
process.load("Validation.RecoTrack.SiTrackingRecHitsValid_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")
process.load("FastSimulation.TrackingRecHitProducer.FamosClusterTest_cff")
process.load("FastSimulation.Configuration.mixNoPU_cfi")

#CPEs
process.load("FastSimulation.TrackingRecHitProducer.FastPixelCPE_cfi")
process.load("FastSimulation.TrackingRecHitProducer.FastStripCPE_cfi")

process.maxEvents = cms.untracked.PSet(
    input = nevents
)
process.source = cms.Source("PoolSource",
    fileNames = inputfiles,
     secondaryFileNames = secinputfiles                          
)

process.PixelTrackingRecHitsValid.src = 'TrackRefitter'
process.StripTrackingRecHitsValid.trajectoryInput = 'TrackRefitter'

process.pixRecHitsValid.outputFile='pixelrechitshisto_310_pre6.root'
process.stripRecHitsValid.outputFile='sistriprechitshisto_310_pre6.root'
process.PixelTrackingRecHitsValid.outputFile='pixeltrackingrechitshist_310_pre6.root'
process.StripTrackingRecHitsValid.outputFile='striptrackingrechitshisto_310_pre6.root'
process.rechits = cms.Sequence(process.famosWithTrackerHits*process.siClusterTranslator*process.siPixelRecHits*process.siStripMatchedRecHits*process.trackerRecHitsValidation)
process.trackinghits = cms.Sequence(process.TrackRefitter*process.trackingRecHitsValid)
process.p1 = cms.Path(process.rechits)
