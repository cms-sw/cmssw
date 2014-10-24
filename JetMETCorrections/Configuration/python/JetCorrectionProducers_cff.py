import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionServices_cff import *

# L1 Correction Producers
ak4CaloJetsL1 = cms.EDProducer(
    'CaloJetCorrectionProducer',
    src        = cms.InputTag('ak4CaloJets'),
    correctors = cms.vstring('L1Fastjet')
    )
ak4PFJetsL1 = cms.EDProducer(
    'PFJetCorrectionProducer',
    src        = cms.InputTag('ak4PFJets'),
    correctors = cms.vstring('L1Fastjet')
    )
ak4PFCHSJetsL1 = cms.EDProducer(
    'PFJetCorrectionProducer',
    src        = cms.InputTag('ak4PFJetsCHS'),
    correctors = cms.vstring('L1Fastjet')
    )

ak4JPTJetsL1 = cms.EDProducer(
    'JPTJetCorrectionProducer',
    src        = cms.InputTag('JetPlusTrackZSPCorJetAntiKt4'),
    correctors = cms.vstring('L1Fastjet')
    )
ak4TrackJetsL1 = cms.EDProducer(
    'TrackJetCorrectionProducer',
    src        = cms.InputTag('ak4TrackJets'),
    correctors = cms.vstring('L1Fastjet')
    )



# L2 Correction Producers
ak4CaloJetsL2 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL2Relative'])
ak4PFJetsL2 = ak4PFJetsL1.clone(correctors = ['ak4PFL2Relative'])
ak4PFCHSJetsL2 = ak4PFCHSJetsL1.clone(correctors = ['ak4PFCHSL2Relative'])
ak4JPTJetsL2 = ak4JPTJetsL1.clone(correctors = ['ak4JPTL2Relative'])
ak4TrackJetsL2 = ak4TrackJetsL1.clone(correctors = ['ak5TRKL2Relative'])

# L2L3 Correction Producers
ak4CaloJetsL2L3 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL2L3'])
ak4PFJetsL2L3 = ak4PFJetsL1.clone(correctors = ['ak4PFL2L3'])
ak4PFCHSJetsL2L3 = ak4PFCHSJetsL1.clone(correctors = ['ak4PFCHSL2L3'])
ak4JPTJetsL2L3 = ak4JPTJetsL1.clone(correctors = ['ak4JPTL2L3'])
ak4TrackJetsL2L3 = ak4TrackJetsL1.clone(correctors = ['ak5TRKL2L3'])


# L1L2L3 Correction Producers
ak4CaloJetsL1L2L3 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL1L2L3'])
ak4PFJetsL1L2L3 = ak4PFJetsL1.clone(correctors = ['ak4PFL1L2L3'])
ak4PFCHSJetsL1L2L3 = ak4PFCHSJetsL1.clone(correctors = ['ak4PFCHSL1L2L3'])
ak4JPTJetsL1L2L3 = ak4JPTJetsL1.clone(correctors = ['ak4JPTL1L2L3'])
ak4TrackJetsL1L2L3 = ak4TrackJetsL1.clone(correctors = ['ak5TRKL1L2L3'])

# L2L3L6 CORRECTION PRODUCERS
ak4CaloJetsL2L3L6 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL2L3L6'])
ak4PFJetsL2L3L6 = ak4PFJetsL1.clone(correctors = ['ak4PFL2L3L6'])


# L1L2L3L6 CORRECTION PRODUCERS
ak4CaloJetsL1L2L3L6 = ak4CaloJetsL1.clone(correctors = ['ak4CaloL1L2L3L6'])
ak4PFJetsL1L2L3L6 = ak4PFJetsL1.clone(correctors = ['ak4PFL1L2L3L6'])

