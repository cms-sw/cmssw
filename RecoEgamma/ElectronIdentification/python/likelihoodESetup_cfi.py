import FWCore.ParameterSet.Config as cms

ElectronLikelihoodESSource = cms.ESProducer("ElectronLikelihoodESSource",

    useEoverP = cms.bool(False),
    useDeltaEta = cms.bool(True),
    useDeltaPhi = cms.bool(True),
    useHoverE = cms.bool(False),
    useSigmaEtaEta = cms.bool(True),
    useSigmaPhiPhi = cms.bool(True),
    useFBrem = cms.bool(True),                                              
    useOneOverEMinusOneOverP = cms.bool(True),

    signalWeightSplitting = cms.string('class'),
    backgroundWeightSplitting = cms.string('class'),

    splitSignalPdfs = cms.bool(True),
    splitBackgroundPdfs = cms.bool(True)

)


