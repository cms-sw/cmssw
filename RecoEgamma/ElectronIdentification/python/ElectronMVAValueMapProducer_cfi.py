import FWCore.ParameterSet.Config as cms

mvaConfigsForProducer = cms.VPSet( )

# Import and add all desired MVAs
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_PHYS14_PU20bx25_nonTrig_V1_cff import *
mvaConfigsForProducer.append( mvaEleID_PHYS14_PU20bx25_nonTrig_V1_producer_config )

electronMVAValueMapProducer = cms.EDProducer('ElectronMVAValueMapProducer',
                                             # The module automatically detects AOD vs miniAOD, so we configure both
                                             #
                                             # AOD case
                                             #
                                             src = cms.InputTag('gedGsfElectrons'),
                                             #
                                             # miniAOD case
                                             #
                                             srcMiniAOD = cms.InputTag('slimmedElectrons'),
                                             #
                                             # MVA configurations
                                             #
                                             mvaConfigurations = mvaConfigsForProducer
                                             )
