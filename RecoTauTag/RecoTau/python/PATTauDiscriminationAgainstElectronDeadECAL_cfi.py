import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

patTauDiscriminationAgainstElectronDeadECAL = cms.EDProducer(
    "PATTauDiscriminationAgainstElectronDeadECAL",

    # tau collection to discriminate
    PATTauProducer = cms.InputTag('slimmedTaus'),

    # require no specific prediscriminants when running on MiniAOD,
    # assuming that loose tau decay mode (and hence leading track) selection is already applied during MiniAOD production
    Prediscriminants = noPrediscriminants,

    # status flag indicating dead/masked ECAL crystals
    minStatus = cms.uint32(12),

    # region around dead/masked ECAL crystals that is to be cut
    dR = cms.double(0.08),

    # extrapolate leading track to ECAL or use tau direction at the primary event vertex
    # in order to compute distance between tau and ECAL crystals
    extrapolateToECalEntrance = cms.bool(True),

    verbosity = cms.int32(0)
)
