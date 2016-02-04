import FWCore.ParameterSet.Config as cms

# creates the recoGsfTracks_electronGsfTracks__RECO = input GSF tracks
from TrackingTools.GsfTracking.GsfElectronTracking_cff import *
ecalDrivenElectronSeeds.SeedConfiguration.initialSeeds = "hiPixelTrackSeeds"
electronCkfTrackCandidates.src = "ecalDrivenElectronSeeds"

electronGsfTrackingHi = cms.Sequence(ecalDrivenElectronSeeds *
                                     electronCkfTrackCandidates *
                                     electronGsfTracks)

# run the supercluster(EE+EB)-GSF track association ==> output: recoGsfElectrons_gsfElectrons__RECO  
from RecoEgamma.EgammaElectronProducers.gsfElectronSequence_cff import *
from RecoParticleFlow.PFProducer.pfElectronTranslator_cff import *
gsfElectrons.ctfTracks     = cms.InputTag("hiGlobalPrimTracks")
gsfElectronCores.ctfTracks = cms.InputTag("hiGlobalPrimTracks")

hiElectronSequence = cms.Sequence(electronGsfTrackingHi * 
			          pfElectronTranslator * 
                                  gsfElectronSequence)
