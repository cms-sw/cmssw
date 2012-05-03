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
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v4_NoEleMatch_Barrel_BDT.weights.xml'),
    inputFileName1prongBL                     = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v4_woG_Barrel_BDT.weights.xml'),
    inputFileName1prongStripsWOgsfBL          = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v4_wGwoGSF_Barrel_BDT.weights.xml'),
    inputFileName1prongStripsWgsfWOpfEleMvaBL = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v4_wGwGSFwoPFMVA_Barrel_BDT.weights.xml'),
    inputFileName1prongStripsWgsfWpfEleMvaBL  = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v4_wGwGSFwPFMVA_Barrel_BDT.weights.xml'),
    inputFileName1prongNoEleMatchEC           = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v4_NoEleMatch_Endcap_BDT.weights.xml'),
    inputFileName1prongEC                     = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v4_woG_Endcap_BDT.weights.xml'),
    inputFileName1prongStripsWOgsfEC          = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v4_wGwoGSF_Endcap_BDT.weights.xml'),
    inputFileName1prongStripsWgsfWOpfEleMvaEC = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v4_wGwGSFwoPFMVA_Endcap_BDT.weights.xml'),
    inputFileName1prongStripsWgsfWpfEleMvaEC  = \
      cms.FileInPath('RecoTauTag/RecoTau/data/TMVAClassification_v4_wGwGSFwPFMVA_Endcap_BDT.weights.xml'),

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
