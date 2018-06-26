import FWCore.ParameterSet.Config as cms

mvaConfigsForEleProducer = cms.VPSet( )

# Import and add all desired MVAs
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_HZZ_V1_cff \
    import mvaEleID_Spring16_HZZ_V1_producer_config
mvaConfigsForEleProducer.append( mvaEleID_Spring16_HZZ_V1_producer_config )

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff \
    import mvaEleID_Spring16_GeneralPurpose_V1_producer_config
mvaConfigsForEleProducer.append( mvaEleID_Spring16_GeneralPurpose_V1_producer_config )

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V1_cff \
    import mvaEleID_Fall17_noIso_V1_producer_config
mvaConfigsForEleProducer.append( mvaEleID_Fall17_noIso_V1_producer_config )

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V1_cff \
    import mvaEleID_Fall17_iso_V1_producer_config
mvaConfigsForEleProducer.append( mvaEleID_Fall17_iso_V1_producer_config )

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_noIso_V2_cff \
    import mvaEleID_Fall17_noIso_V2_producer_config
mvaConfigsForEleProducer.append( mvaEleID_Fall17_noIso_V2_producer_config )

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Fall17_iso_V2_cff \
    import mvaEleID_Fall17_iso_V2_producer_config
mvaConfigsForEleProducer.append( mvaEleID_Fall17_iso_V2_producer_config )

# The producer to compute the MVA input variables which are not accessible with the cut parser
electronMVAVariableHelper = cms.EDProducer('GsfElectronMVAVariableHelper',
                                             # The module automatically detects AOD vs miniAOD, so we configure both
                                             # AOD case
                                             src = cms.InputTag('gedGsfElectrons'),
                                             vertexCollection = cms.InputTag("offlinePrimaryVertices"),
                                             beamSpot         = cms.InputTag("offlineBeamSpot"),
                                             conversions      = cms.InputTag("allConversions"),
                                             # miniAOD case
                                             srcMiniAOD              = cms.InputTag('slimmedElectrons',processName=cms.InputTag.skipCurrentProcess()),
                                             vertexCollectionMiniAOD = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                             beamSpotMiniAOD         = cms.InputTag("offlineBeamSpot"),
                                             conversionsMiniAOD      = cms.InputTag("reducedEgamma:reducedConversions"),
                                             )

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
