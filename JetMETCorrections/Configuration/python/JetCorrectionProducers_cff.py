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


# L2L3 Correction Producers
ak5CaloJetsL2 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL2Relative'])
ak5PFJetsL2 = ak5PFJetsL1.clone(correctors = ['ak5PFL2Relative'])


# L2L3 Correction Producers
ak5CaloJetsL2L3 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL2L3'])
ak5PFJetsL2L3 = ak5PFJetsL1.clone(correctors = ['ak5PFL2L3'])


# L1L2L3 Correction Producers
ak5CaloJetsL1L2L3 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL1L2L3'])
ak5PFJetsL1L2L3 = ak5PFJetsL1.clone(correctors = ['ak5PFL1L2L3'])


# L2L3L6 CORRECTION PRODUCERS
ak5CaloJetsL2L3L6 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL2L3L6'])
ak5PFJetsL2L3L6 = ak5PFJetsL1.clone(correctors = ['ak5PFL2L3L6'])


# L1L2L3L6 CORRECTION PRODUCERS
ak5CaloJetsL1L2L3L6 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL1L2L3L6'])
ak5PFJetsL1L2L3L6 = ak5PFJetsL1.clone(correctors = ['ak5PFL1L2L3L6'])
