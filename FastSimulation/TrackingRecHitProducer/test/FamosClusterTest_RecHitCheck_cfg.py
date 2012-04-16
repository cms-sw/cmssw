#Config

import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackerValidationOnly")

# The number of events to be processed.
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.StandardSequences.Services_cff")
# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

process.load("FastSimulation/Configuration/CommonInputs_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "IDEAL_31X::All"
process.load("FastSimulation/Configuration/FamosSequences_cff")
process.load("FastSimulation/TrackingRecHitProducer/FamosClusterTest_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = gtag
#process.load("SimTracker.Configuration.SimTracker_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("RecoTracker.Configuration.RecoTracker_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.load("Validation.TrackerRecHits.trackerRecHitsValidation_cff")
process.load("Validation.RecoTrack.SiTrackingRecHitsValid_cff")


#CPEs
process.load("FastSimulation.TrackingRecHitProducer.FastPixelCPE_cfi")
process.load("FastSimulation.TrackingRecHitProducer.FastStripCPE_cfi")

#process.maxEvents = cms.untracked.PSet(
#    input = nevents
#    )
#process.source = cms.Source("PoolSource",
#                            fileNames = inputfiles,
#     secondaryFileNames = secinputfiles                          
 #                           )

process.source = cms.Source("PoolSource",
                            debugFlag = cms.untracked.bool(True),
                            debugVebosity = cms.untracked.uint32(10),
                            fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0004/B89FA4AB-CC41-DE11-8348-000423D98B28.root',
    '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0004/58AEC160-E741-DE11-AE53-001D09F2AF1E.root',
    '/store/relval/CMSSW_3_1_0_pre7/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_31X_v1/0004/50F087DB-CB41-DE11-98F2-0030487C5CFA.root'
    )
                            )


process.PixelTrackingRecHitsValid.src = 'TrackRefitter'
process.StripTrackingRecHitsValid.trajectoryInput = 'TrackRefitter'



process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.destinations = ['detailedInfoRecHitVal.txt']

process.pixRecHitsValid.outputFile='pixelrechitshisto_310_pre7.root'
process.stripRecHitsValid.outputFile='sistriprechitshisto_310_pre7.root'
process.PixelTrackingRecHitsValid.outputFile='pixeltrackingrechitshist_310_pre7.root'
process.StripTrackingRecHitsValid.outputFile='striptrackingrechitshisto_310_pre7.root'
process.trackinghits = cms.Sequence(process.TrackRefitter*process.trackingRecHitsValid)
process.rechits = cms.Sequence(process.famosWithTrackerHits*process.siClusterTranslator*process.siPixelRecHits*process.siStripMatchedRecHits*process.trackerRecHitsValidation*process.ckftracks*process.trackinghits)
process.p1 = cms.Path(process.rechits)
