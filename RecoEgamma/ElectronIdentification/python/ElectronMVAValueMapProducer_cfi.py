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

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_noIso_V1_cff \
    import mvaEleID_RunIIIWinter22_noIso_V1_producer_config
mvaConfigsForEleProducer.append( mvaEleID_RunIIIWinter22_noIso_V1_producer_config )

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_RunIIIWinter22_iso_V1_cff \
    import mvaEleID_RunIIIWinter22_iso_V1_producer_config
mvaConfigsForEleProducer.append( mvaEleID_RunIIIWinter22_iso_V1_producer_config )

# HZZ4l Run2 (Ultra)Legacy 
from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer16UL_ID_ISO_cff \
    import mvaEleID_Summer16UL_ID_ISO_producer_config
mvaConfigsForEleProducer.append(mvaEleID_Summer16UL_ID_ISO_producer_config)

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer17UL_ID_ISO_cff \
    import mvaEleID_Summer17UL_ID_ISO_producer_config
mvaConfigsForEleProducer.append(mvaEleID_Summer17UL_ID_ISO_producer_config)

from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Summer18UL_ID_ISO_cff \
    import mvaEleID_Summer18UL_ID_ISO_producer_config
mvaConfigsForEleProducer.append( mvaEleID_Summer18UL_ID_ISO_producer_config )

electronMVAValueMapProducer = cms.EDProducer('ElectronMVAValueMapProducer',
                                             src = cms.InputTag('slimmedElectrons'),
                                             mvaConfigurations = mvaConfigsForEleProducer
                                             )
