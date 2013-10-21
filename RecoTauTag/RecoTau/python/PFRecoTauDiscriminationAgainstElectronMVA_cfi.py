import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationAgainstElectronMVA = cms.EDProducer(
    "PFRecoTauDiscriminationAgainstElectronMVA",

    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    method = cms.string("BDT"),
    inputFileName1prongBL            = cms.FileInPath("RecoTauTag/RecoTau/data/TMVAClassification_v2_X_0BL_BDT.weights.xml.gz"),
    inputFileName1prongStripsWgsfBL  = cms.FileInPath("RecoTauTag/RecoTau/data/TMVAClassification_v2_1_1BL_BDT.weights.xml.gz"),
    inputFileName1prongStripsWOgsfBL = cms.FileInPath("RecoTauTag/RecoTau/data/TMVAClassification_v2_0_1BL_BDT.weights.xml.gz"),
    inputFileName1prongEC            = cms.FileInPath("RecoTauTag/RecoTau/data/TMVAClassification_v2_X_0EC_BDT.weights.xml.gz"),
    inputFileName1prongStripsWgsfEC  = cms.FileInPath("RecoTauTag/RecoTau/data/TMVAClassification_v2_1_1EC_BDT.weights.xml.gz"),
    inputFileName1prongStripsWOgsfEC = cms.FileInPath("RecoTauTag/RecoTau/data/TMVAClassification_v2_0_1EC_BDT.weights.xml.gz"),

    returnMVA = cms.bool(False),
    minMVA1prongBL            = cms.double(0.054),
    minMVA1prongStripsWgsfBL  = cms.double(0.060),
    minMVA1prongStripsWOgsfBL = cms.double(0.054),
    minMVA1prongEC            = cms.double(0.060),
    minMVA1prongStripsWgsfEC  = cms.double(0.053),
    minMVA1prongStripsWOgsfEC = cms.double(0.049),
)
