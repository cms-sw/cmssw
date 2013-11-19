import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionServices_cff import *

# L1 Correction Producers
ak5CaloJetsL1 = cms.EDProducer(
    'CaloJetCorrectionProducer',
    src        = cms.InputTag('ak5CaloJets'),
    correctors = cms.vstring('L1Fastjet')
    )
ak5PFJetsL1 = cms.EDProducer(
    'PFJetCorrectionProducer',
    src        = cms.InputTag('ak5PFJets'),
    correctors = cms.vstring('L1Fastjet')
    )
ak5PFCHSJetsL1 = cms.EDProducer(
    'PFJetCorrectionProducer',
    src        = cms.InputTag('ak5PFJetsCHS'),
    correctors = cms.vstring('L1Fastjet')
    )

ak5JPTJetsL1 = cms.EDProducer(
    'JPTJetCorrectionProducer',
    src        = cms.InputTag('JetPlusTrackZSPCorJetAntiKt5'),
    correctors = cms.vstring('L1Fastjet')
    )
ak5TrackJetsL1 = cms.EDProducer(
    'TrackJetCorrectionProducer',
    src        = cms.InputTag('ak5TrackJets'),
    correctors = cms.vstring('L1Fastjet')
    )



# L2L3 Correction Producers
ak5CaloJetsL2 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL2Relative'])
ak5PFJetsL2 = ak5PFJetsL1.clone(correctors = ['ak5PFL2Relative'])
ak5PFCHSJetsL2 = ak5PFCHSJetsL1.clone(correctors = ['ak5PFCHSL2Relative'])
ak5JPTJetsL2 = ak5JPTJetsL1.clone(correctors = ['ak5JPTL2Relative'])
ak5TrackJetsL2 = ak5TrackJetsL1.clone(correctors = ['ak5TRKL2Relative'])

# L2L3 Correction Producers
ak5CaloJetsL2L3 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL2L3'])
ak5PFJetsL2L3 = ak5PFJetsL1.clone(correctors = ['ak5PFL2L3'])
ak5PFCHSJetsL2L3 = ak5PFCHSJetsL1.clone(correctors = ['ak5PFCHSL2L3'])
ak5JPTJetsL2L3 = ak5JPTJetsL1.clone(correctors = ['ak5JPTL2L3'])
ak5TrackJetsL2L3 = ak5TrackJetsL1.clone(correctors = ['ak5TRKL2L3'])


# L1L2L3 Correction Producers
ak5CaloJetsL1L2L3 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL1L2L3'])
ak5PFJetsL1L2L3 = ak5PFJetsL1.clone(correctors = ['ak5PFL1L2L3'])
ak5PFCHSJetsL1L2L3 = ak5PFCHSJetsL1.clone(correctors = ['ak5PFCHSL1L2L3'])
ak5JPTJetsL1L2L3 = ak5JPTJetsL1.clone(correctors = ['ak5JPTL1L2L3'])
ak5TrackJetsL1L2L3 = ak5TrackJetsL1.clone(correctors = ['ak5TRKL1L2L3'])

# L2L3L6 CORRECTION PRODUCERS
ak5CaloJetsL2L3L6 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL2L3L6'])
ak5PFJetsL2L3L6 = ak5PFJetsL1.clone(correctors = ['ak5PFL2L3L6'])


# L1L2L3L6 CORRECTION PRODUCERS
ak5CaloJetsL1L2L3L6 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL1L2L3L6'])
ak5PFJetsL1L2L3L6 = ak5PFJetsL1.clone(correctors = ['ak5PFL1L2L3L6'])

