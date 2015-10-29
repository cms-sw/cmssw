import FWCore.ParameterSet.Config as cms

mvaConfigsForEleProducer = cms.VPSet( )

# Import and add all desired MVAs
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_PHYS14_PU20bx25_nonTrig_V1_cff import *
mvaConfigsForEleProducer.append( mvaEleID_PHYS14_PU20bx25_nonTrig_V1_producer_config )

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_nonTrig_V1_cff import *
mvaConfigsForEleProducer.append( mvaEleID_Spring15_25ns_nonTrig_V1_producer_config )

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_50ns_Trig_V1_cff import *
mvaConfigsForEleProducer.append( mvaEleID_Spring15_50ns_Trig_V1_producer_config )

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_Trig_V1_cff import *
mvaConfigsForEleProducer.append( mvaEleID_Spring15_25ns_Trig_V1_producer_config )

electronMVAValueMapProducer = cms.EDProducer('ElectronMVAValueMapProducer',
                                             # The module automatically detects AOD vs miniAOD, so we configure both
                                             #
                                             # AOD case
                                             #
                                             src = cms.InputTag('gedGsfElectrons'),
                                             #
                                             # miniAOD case
                                             #
                                             srcMiniAOD = cms.InputTag('slimmedElectrons',processName=cms.InputTag.skipCurrentProcess()),
                                             #
                                             # MVA configurations
                                             #
                                             mvaConfigurations = mvaConfigsForEleProducer
                                             )
