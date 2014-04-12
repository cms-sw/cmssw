import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationAgainstElectronMVA5GBR = cms.EDProducer("PFRecoTauDiscriminationAgainstElectronMVA5GBR",

    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    method = cms.string("BDTG"),

    gbrFile = cms.FileInPath('RecoTauTag/RecoTau/data/gbrDiscriminationAgainstElectronMVA5.root'),
    
    returnMVA = cms.bool(True),

    minMVANoEleMatchWOgWOgsfBL = cms.double(0.0),
    minMVANoEleMatchWOgWgsfBL  = cms.double(0.0),
    minMVANoEleMatchWgWOgsfBL  = cms.double(0.0),
    minMVANoEleMatchWgWgsfBL   = cms.double(0.0),
    minMVAWOgWOgsfBL           = cms.double(0.0),
    minMVAWOgWgsfBL            = cms.double(0.0),
    minMVAWgWOgsfBL            = cms.double(0.0),
    minMVAWgWgsfBL             = cms.double(0.0),
    minMVANoEleMatchWOgWOgsfEC = cms.double(0.0),
    minMVANoEleMatchWOgWgsfEC  = cms.double(0.0),
    minMVANoEleMatchWgWOgsfEC  = cms.double(0.0),
    minMVANoEleMatchWgWgsfEC   = cms.double(0.0),
    minMVAWOgWOgsfEC           = cms.double(0.0),
    minMVAWOgWgsfEC            = cms.double(0.0),
    minMVAWgWOgsfEC            = cms.double(0.0),
    minMVAWgWgsfEC             = cms.double(0.0),

    srcGsfElectrons = cms.InputTag('gedGsfElectrons')
)
