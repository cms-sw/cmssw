import FWCore.ParameterSet.Config as cms

# load local PF Reco
from RecoHI.Configuration.Reconstruction_hiPF_cff import HiParticleFlowLocalReco

# creates the recoGsfTracks_electronGsfTracks__RECO = input GSF tracks
from TrackingTools.GsfTracking.GsfElectronTracking_cff import *
ecalDrivenElectronSeeds.SeedConfiguration.initialSeeds = "hiPixelTrackSeeds"
electronCkfTrackCandidates.src = "ecalDrivenElectronSeeds"

ecalDrivenElectronSeeds.barrelSuperClusters = cms.InputTag("correctedIslandBarrelSuperClusters")

ecalDrivenElectronSeeds.SeedConfiguration.maxHOverEBarrel = cms.double(0.25)
ecalDrivenElectronSeeds.SeedConfiguration.maxHOverEEndcaps = cms.double(0.25)

electronGsfTrackingHi = cms.Sequence(ecalDrivenElectronSeeds *
                                     electronCkfTrackCandidates *
                                     electronGsfTracks)

# run the supercluster(EE+EB)-GSF track association ==> output: recoGsfElectrons_gsfElectrons__RECO
from RecoEgamma.EgammaElectronProducers.gsfElectronSequence_cff import *
from RecoParticleFlow.PFProducer.pfElectronTranslator_cff import *
gsfElectrons.ctfTracks     = cms.InputTag("hiSelectedTracks")
gsfElectronCores.ctfTracks = cms.InputTag("hiSelectedTracks")
pfElectronTranslator.emptyIsOk = cms.bool(True)

ecalDrivenGsfElectrons.ctfTracksTag = cms.InputTag("hiSelectedTracks")
ecalDrivenGsfElectronCores.ctfTracks = cms.InputTag("hiSelectedTracks")

ecalDrivenGsfElectrons.maxHOverEBarrel = cms.double(0.25)
ecalDrivenGsfElectrons.maxHOverEEndcaps = cms.double(0.25)

hiElectronSequence = cms.Sequence(electronGsfTrackingHi *
                                  HiParticleFlowLocalReco *
                                  gsfEcalDrivenElectronSequence
                                  )
