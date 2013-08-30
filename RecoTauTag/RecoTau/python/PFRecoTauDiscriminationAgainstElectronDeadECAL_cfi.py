import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationAgainstElectronDeadECAL = cms.EDProducer(
    "PFRecoTauDiscriminationAgainstElectronDeadECAL",

    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    # status flag indicating dead/masked ECAL crystals
    minStatus = cms.uint32(12),

    # region around dead/masked ECAL crystals that is to be cut                                                               
    dR = cms.double(0.08)
)
