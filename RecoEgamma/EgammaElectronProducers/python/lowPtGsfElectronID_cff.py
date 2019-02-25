import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.defaultLowPtGsfElectronID_cfi import defaultLowPtGsfElectronID

lowPtGsfElectronID = defaultLowPtGsfElectronID.clone(
    ModelNames = cms.vstring(['']),
    ModelWeights = cms.vstring([
            'RecoEgamma/ElectronIdentification/data/LowPtElectrons/RunII_Fall17_LowPtElectrons_mva_id.xml.gz',
            ]),
    ModelThresholds = cms.vdouble([-10.])
    )
