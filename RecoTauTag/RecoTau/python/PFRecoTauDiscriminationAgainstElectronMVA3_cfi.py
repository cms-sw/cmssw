import FWCore.ParameterSet.Config as cms
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack

pfRecoTauDiscriminationAgainstElectronMVA3 = cms.EDProducer(
    "PFRecoTauDiscriminationAgainstElectronMVA3",

    # tau collection to discriminate
    PFTauProducer = cms.InputTag('pfTauProducer'),

    # Require leading pion ensures that:
    #  1) these is at least one track above threshold (0.5 GeV) in the signal cone
    #  2) a track OR a pi-zero in the signal cone has pT > 5 GeV
    Prediscriminants = requireLeadTrack,

    method = cms.string("BDTG"),

    inputFileName1prongNoEleMatchWOgWOgsfBL = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_NoEleMatch_woGwoGSF_Barrel_BDTG.weights.xml.gz'),
    inputFileName1prongNoEleMatchWOgWgsfBL  = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_NoEleMatch_woGwGSF_Barrel_BDTG.weights.xml.gz'),
    inputFileName1prongNoEleMatchWgWOgsfBL  = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_NoEleMatch_wGwoGSF_Barrel_BDTG.weights.xml.gz'),
    inputFileName1prongNoEleMatchWgWgsfBL   = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_NoEleMatch_wGwGSF_Barrel_BDTG.weights.xml.gz'),
    inputFileName1prongWOgWOgsfBL           = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_woGwoGSF_Barrel_BDTG.weights.xml.gz'),
    inputFileName1prongWOgWgsfBL            = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_woGwGSF_Barrel_BDTG.weights.xml.gz'),
    inputFileName1prongWgWOgsfBL            = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_wGwoGSF_Barrel_BDTG.weights.xml.gz'),
    inputFileName1prongWgWgsfBL             = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_wGwGSF_Barrel_BDTG.weights.xml.gz'),
    inputFileName1prongNoEleMatchWOgWOgsfEC = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_NoEleMatch_woGwoGSF_Endcap_BDTG.weights.xml.gz'),
    inputFileName1prongNoEleMatchWOgWgsfEC  = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_NoEleMatch_woGwGSF_Endcap_BDTG.weights.xml.gz'),
    inputFileName1prongNoEleMatchWgWOgsfEC  = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_NoEleMatch_wGwoGSF_Endcap_BDTG.weights.xml.gz'),
    inputFileName1prongNoEleMatchWgWgsfEC   = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_NoEleMatch_wGwGSF_Endcap_BDTG.weights.xml.gz'),
    inputFileName1prongWOgWOgsfEC           = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_woGwoGSF_Endcap_BDTG.weights.xml.gz'),
    inputFileName1prongWOgWgsfEC            = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_woGwGSF_Endcap_BDTG.weights.xml.gz'),
    inputFileName1prongWgWOgsfEC            = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_wGwoGSF_Endcap_BDTG.weights.xml.gz'),
    inputFileName1prongWgWgsfEC             = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v13EleVeto_wGwGSF_Endcap_BDTG.weights.xml.gz'),

    
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
