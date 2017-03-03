import FWCore.ParameterSet.Config as cms

mvaConfigsForPhoProducer = cms.VPSet( )

# Import and add all desired MVAs

from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_50ns_nonTrig_V2p1_cff import *
mvaConfigsForPhoProducer.append( mvaPhoID_Spring15_50ns_nonTrig_V2p1_producer_config )

from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_25ns_nonTrig_V2p1_cff import *
mvaConfigsForPhoProducer.append( mvaPhoID_Spring15_25ns_nonTrig_V2p1_producer_config )

from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring16_nonTrig_V1_cff \
    import mvaPhoID_Spring16_nonTrig_V1_producer_config
mvaConfigsForPhoProducer.append( mvaPhoID_Spring16_nonTrig_V1_producer_config )

photonMVAValueMapProducer = cms.EDProducer('PhotonMVAValueMapProducer',
                                           # The module automatically detects AOD vs miniAOD, so we configure both
                                           #
                                           # AOD case
                                           #
                                           src = cms.InputTag('gedPhotons'),
                                           #
                                           # miniAOD case
                                           #
                                           srcMiniAOD = cms.InputTag('slimmedPhotons',processName=cms.InputTag.skipCurrentProcess()),
                                           #
                                           # MVA configurations
                                           #
                                           mvaConfigurations = mvaConfigsForPhoProducer
                                           )
