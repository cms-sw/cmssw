import FWCore.ParameterSet.Config as cms

lowPtGsfElectronID = cms.EDProducer(
    "LowPtGsfElectronIDProducer",
    electrons = cms.InputTag("lowPtGsfElectrons"),
    rho = cms.InputTag("fixedGridRhoFastjetAllTmp"),
    ModelNames = cms.vstring(['']), # default is empty string
    ModelWeights = cms.vstring(['RecoEgamma/ElectronIdentification/data/LowPtElectrons/RunII_Fall17_LowPtElectrons_mva_id.xml.gz']),
    ModelThresholds = cms.vdouble([-1.]),
    PassThrough = cms.bool(False),
    MinPtThreshold = cms.double(0.5),
    MaxPtThreshold = cms.double(15.),
    )
