import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationAgainstElectronMVA3GBR = cms.EDProducer(
    "PFRecoTauDiscriminationAgainstElectronMVA3GBR",

    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    method = cms.string("BDTG"),

    gbrFile = \
      cms.FileInPath('RecoTauTag/RecoTau/data/GBRForest_v13EleVeto_BDTG.root'),
    
    returnMVA = cms.bool(True),

    minMVA1prongNoEleMatchWOgWOgsfBL   = cms.double(0.0),
    minMVA1prongNoEleMatchWOgWgsfBL    = cms.double(0.0),
    minMVA1prongNoEleMatchWgWOgsfBL    = cms.double(0.0),
    minMVA1prongNoEleMatchWgWgsfBL     = cms.double(0.0),
    minMVA1prongWOgWOgsfBL             = cms.double(0.0),
    minMVA1prongWOgWgsfBL              = cms.double(0.0),
    minMVA1prongWgWOgsfBL              = cms.double(0.0),
    minMVA1prongWgWgsfBL               = cms.double(0.0),
    minMVA1prongNoEleMatchWOgWOgsfEC   = cms.double(0.0),
    minMVA1prongNoEleMatchWOgWgsfEC    = cms.double(0.0),
    minMVA1prongNoEleMatchWgWOgsfEC    = cms.double(0.0),
    minMVA1prongNoEleMatchWgWgsfEC     = cms.double(0.0),
    minMVA1prongWOgWOgsfEC             = cms.double(0.0),
    minMVA1prongWOgWgsfEC              = cms.double(0.0),
    minMVA1prongWgWOgsfEC              = cms.double(0.0),
    minMVA1prongWgWgsfEC               = cms.double(0.0),
    minMVA3prongMatch                  = cms.double(0.0),
    minMVA3prongNoMatch                = cms.double(0.0),

    srcGsfElectrons = cms.InputTag('gsfElectrons')
)
