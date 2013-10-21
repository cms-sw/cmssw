import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationAgainstElectronMVA2 = cms.EDProducer(
    "PFRecoTauDiscriminationAgainstElectronMVA2",

    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    method = cms.string("BDT"),

    inputFileName1prongNoEleMatchBL           = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v5_NoEleMatch_Barrel_BDTG.weights.xml.gz'),
    inputFileName1prongBL                     = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v5_woG_Barrel_BDTG.weights.xml.gz'),
    inputFileName1prongStripsWOgsfBL          = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v5_wGwoGSF_Barrel_BDTG.weights.xml.gz'),
    inputFileName1prongStripsWgsfWOpfEleMvaBL = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v5_wGwGSFwoPFMVA_Barrel_BDTG.weights.xml.gz'),
    inputFileName1prongStripsWgsfWpfEleMvaBL  = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v5_wGwGSFwPFMVA_Barrel_BDTG.weights.xml.gz'),

    inputFileName1prongNoEleMatchEC           = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v5_NoEleMatch_Endcap_BDTG.weights.xml.gz'),
    inputFileName1prongEC                     = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v5_woG_Endcap_BDTG.weights.xml.gz'),
    inputFileName1prongStripsWOgsfEC          = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v5_wGwoGSF_Endcap_BDTG.weights.xml.gz'),
    inputFileName1prongStripsWgsfWOpfEleMvaEC = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v5_wGwGSFwoPFMVA_Endcap_BDTG.weights.xml.gz'),
    inputFileName1prongStripsWgsfWpfEleMvaEC  = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v5_wGwGSFwPFMVA_Endcap_BDTG.weights.xml.gz'),

    returnMVA = cms.bool(True),

    minMVA1prongNoEleMatchBL           = cms.double(0.0),
    minMVA1prongBL                     = cms.double(0.0),
    minMVA1prongStripsWOgsfBL          = cms.double(0.0),
    minMVA1prongStripsWgsfWOpfEleMvaBL = cms.double(0.0),
    minMVA1prongStripsWgsfWpfEleMvaBL  = cms.double(0.0),
    minMVA1prongNoEleMatchEC           = cms.double(0.0),
    minMVA1prongEC                     = cms.double(0.0),
    minMVA1prongStripsWOgsfEC          = cms.double(0.0),
    minMVA1prongStripsWgsfWOpfEleMvaEC = cms.double(0.0),
    minMVA1prongStripsWgsfWpfEleMvaEC  = cms.double(0.0),

    srcGsfElectrons = cms.InputTag('gsfElectrons')
)
