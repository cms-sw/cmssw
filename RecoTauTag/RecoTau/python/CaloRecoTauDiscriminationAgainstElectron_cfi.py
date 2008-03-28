import FWCore.ParameterSet.Config as cms

caloRecoTauDiscriminationAgainstElectron = cms.EDFilter("CaloRecoTauDiscriminationAgainstElectron",
    ApplyCut_maxleadTrackHCAL3x3hottesthitDEta = cms.bool(False),
    CaloTauProducer = cms.string('caloRecoTauProducer'),
    leadTrack_HCAL3x3hitsEtSumOverPt_minvalue = cms.double(0.1),
    ApplyCut_leadTrackavoidsECALcrack = cms.bool(True),
    maxleadTrackHCAL3x3hottesthitDEta = cms.double(0.1)
)


