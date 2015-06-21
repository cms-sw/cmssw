import FWCore.ParameterSet.Config as cms

mvaConfigsForProducer = cms.VPSet( )

# Import and add all desired MVAs
from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_PHYS14_PU20bx25_nonTrig_V1_cff import *
mvaConfigsForProducer.append( mvaPhoID_PHYS14_PU20bx25_nonTrig_V1_producer_config )

photonMVAValueMapProducer = cms.EDProducer('PhotonMVAValueMapProducer',
                                           # The module automatically detects AOD vs miniAOD, so we configure both
                                           #
                                           # AOD case
                                           #
                                           src = cms.InputTag('gedPhotons'),
                                           #
                                           # miniAOD case
                                           #
                                           srcMiniAOD = cms.InputTag('slimmedPhotons'),
                                           #
                                           # MVA configurations
                                           #
                                           mvaConfigurations = mvaConfigsForProducer
                                           )
