import FWCore.ParameterSet.Config as cms


# creates the recoGsfTracks_electronGsfTracks__RECO = input GSF tracks
from TrackingTools.GsfTracking.GsfElectronTracking_cff import *
ecalDrivenElectronSeeds.initialSeedsVector = ["hiPixelTrackSeeds"]
electronCkfTrackCandidates.src = "ecalDrivenElectronSeeds"

ecalDrivenElectronSeeds.maxHOverEBarrel = 0.25
ecalDrivenElectronSeeds.maxHOverEEndcaps = 0.25

electronGsfTrackingHiTask = cms.Task(ecalDrivenElectronSeeds ,
                                     electronCkfTrackCandidates ,
                                     electronGsfTracks)

from RecoEgamma.EgammaElectronProducers.gsfElectronSequence_cff import *

ecalDrivenGsfElectrons.ctfTracksTag = "hiGeneralTracks"
ecalDrivenGsfElectronCores.ctfTracks = "hiGeneralTracks"
ecalDrivenGsfElectrons.vtxTag = "hiSelectedVertex"

ecalDrivenGsfElectrons.preselection.maxHOverEBarrelCone = 0.25
ecalDrivenGsfElectrons.preselection.maxHOverEEndcapsCone = 0.25
ecalDrivenGsfElectrons.preselection.maxHOverEBarrelBc = 0.
ecalDrivenGsfElectrons.preselection.maxHOverEEndcapsBc = 0.
ecalDrivenGsfElectrons.fillConvVtxFitProb = False


from RecoParticleFlow.PFTracking.pfTrack_cfi import *
pfTrack.UseQuality = True
pfTrack.TrackQuality = 'highPurity'
pfTrack.TkColList = ["hiGeneralTracks"]
pfTrack.PrimaryVertexLabel = "hiSelectedVertex"
pfTrack.MuColl = "hiMuons1stStep"

from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
pfTrackElec.applyGsfTrackCleaning = True
pfTrackElec.PrimaryVertexLabel = "hiSelectedVertex"

hiElectronTask = cms.Task(electronGsfTrackingHiTask ,   
                          pfTrack ,
                          pfTrackElec ,
                          gsfEcalDrivenElectronTask 
                          )
hiElectronSequence = cms.Sequence(hiElectronTask)
