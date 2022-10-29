import FWCore.ParameterSet.Config as cms

mvaConfigsForPhoProducer = cms.VPSet( )

# Import and add all desired MVAs

from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring16_nonTrig_V1_cff \
    import mvaPhoID_Spring16_nonTrig_V1_producer_config
mvaConfigsForPhoProducer.append( mvaPhoID_Spring16_nonTrig_V1_producer_config )

from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V1p1_cff import *
mvaConfigsForPhoProducer.append( mvaPhoID_RunIIFall17_v1p1_producer_config )

#### Run2 ID Fall17
from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Fall17_94X_V2_cff import *
mvaConfigsForPhoProducer.append( mvaPhoID_RunIIFall17_v2_producer_config )

#### Run3 ID Winter22 
from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Winter22_122X_V1_cff import *
mvaConfigsForPhoProducer.append( mvaPhoID_RunIIIWinter22_v1_producer_config )

###Phase II ID valid for EB 
from RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Summer20_PhaseII_V0_cff import *
mvaConfigsForPhoProducer.append( mvaPhoID_PhaseIISummer20_v0_producer_config )

photonMVAValueMapProducer = cms.EDProducer('PhotonMVAValueMapProducer',
                                           src = cms.InputTag('slimmedPhotons'),
                                           mvaConfigurations = mvaConfigsForPhoProducer
                                           )
