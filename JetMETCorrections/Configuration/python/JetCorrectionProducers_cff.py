import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionServices_cff import *

# L1 Correction Producers
ak5CaloJetsL1 = cms.EDProducer(
    'CaloJetCorrectionProducer',
    src        = cms.InputTag('ak5CaloJets'),
    correctors = cms.vstring('L1Fastjet')
    )
ak7CaloJetsL1 = ak5CaloJetsL1.clone( src = 'ak7CaloJets' )
kt4CaloJetsL1 = ak5CaloJetsL1.clone( src = 'kt4CaloJets' )
kt6CaloJetsL1 = ak5CaloJetsL1.clone( src = 'kt6CaloJets' )
sc5CaloJetsL1 = ak5CaloJetsL1.clone( src = 'sc5CaloJets' )
sc7CaloJetsL1 = ak5CaloJetsL1.clone( src = 'sc7CaloJets' )
ic5CaloJetsL1 = ak5CaloJetsL1.clone( src = 'ic5CaloJets' )

ak5PFJetsL1 = cms.EDProducer(
    'PFJetCorrectionProducer',
    src        = cms.InputTag('ak5PFJets'),
    correctors = cms.vstring('L1Fastjet')
    )
ak7PFJetsL1 = ak5PFJetsL1.clone( src = 'ak7PFJets' )
kt4PFJetsL1 = ak5PFJetsL1.clone( src = 'kt4PFJets' )
kt6PFJetsL1 = ak5PFJetsL1.clone( src = 'kt6PFJets' )
sc5PFJetsL1 = ak5PFJetsL1.clone( src = 'sc5PFJets' )
sc7PFJetsL1 = ak5PFJetsL1.clone( src = 'sc7PFJets' )
ic5PFJetsL1 = ak5PFJetsL1.clone( src = 'ic5PFJets' )


# L2L3 Correction Producers
ak5CaloJetsL2 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL2Relative'])
ak7CaloJetsL2 = ak7CaloJetsL1.clone(correctors = ['ak7CaloL2Relative'])
kt4CaloJetsL2 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL2Relative'])
kt6CaloJetsL2 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL2Relative'])
sc5CaloJetsL2 = sc5CaloJetsL1.clone(correctors = ['sc5CaloL2Relative'])
sc7CaloJetsL2 = sc7CaloJetsL1.clone(correctors = ['sc7CaloL2Relative'])
ic5CaloJetsL2 = ic5CaloJetsL1.clone(correctors = ['ic5CaloL2Relative'])

ak5PFJetsL2 = ak5PFJetsL1.clone(correctors = ['ak5PFL2Relative'])
ak7PFJetsL2 = ak7PFJetsL1.clone(correctors = ['ak7PFL2Relative'])
kt4PFJetsL2 = kt4PFJetsL1.clone(correctors = ['kt4PFL2Relative'])
kt6PFJetsL2 = kt6PFJetsL1.clone(correctors = ['kt6PFL2Relative'])
sc5PFJetsL2 = sc5PFJetsL1.clone(correctors = ['sc5PFL2Relative'])
sc7PFJetsL2 = sc7PFJetsL1.clone(correctors = ['sc7PFL2Relative'])
ic5PFJetsL2 = ic5PFJetsL1.clone(correctors = ['ic5PFL2Relative'])


# L2L3 Correction Producers
ak5CaloJetsL2L3 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL2L3'])
ak7CaloJetsL2L3 = ak7CaloJetsL1.clone(correctors = ['ak7CaloL2L3'])
kt4CaloJetsL2L3 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL2L3'])
kt6CaloJetsL2L3 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL2L3'])
sc5CaloJetsL2L3 = sc5CaloJetsL1.clone(correctors = ['sc5CaloL2L3'])
sc7CaloJetsL2L3 = sc7CaloJetsL1.clone(correctors = ['sc7CaloL2L3'])
ic5CaloJetsL2L3 = ic5CaloJetsL1.clone(correctors = ['ic5CaloL2L3'])

ak5PFJetsL2L3 = ak5PFJetsL1.clone(correctors = ['ak5PFL2L3'])
ak7PFJetsL2L3 = ak7PFJetsL1.clone(correctors = ['ak7PFL2L3'])
kt4PFJetsL2L3 = kt4PFJetsL1.clone(correctors = ['kt4PFL2L3'])
kt6PFJetsL2L3 = kt6PFJetsL1.clone(correctors = ['kt6PFL2L3'])
sc5PFJetsL2L3 = sc5PFJetsL1.clone(correctors = ['sc5PFL2L3'])
sc7PFJetsL2L3 = sc7PFJetsL1.clone(correctors = ['sc7PFL2L3'])
ic5PFJetsL2L3 = ic5PFJetsL1.clone(correctors = ['ic5PFL2L3'])


# L1L2L3 Correction Producers
ak5CaloJetsL1L2L3 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL1L2L3'])
ak7CaloJetsL1L2L3 = ak7CaloJetsL1.clone(correctors = ['ak7CaloL1L2L3'])
kt4CaloJetsL1L2L3 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL1L2L3'])
kt6CaloJetsL1L2L3 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL1L2L3'])
sc5CaloJetsL1L2L3 = sc5CaloJetsL1.clone(correctors = ['sc5CaloL1L2L3'])
sc7CaloJetsL1L2L3 = sc7CaloJetsL1.clone(correctors = ['sc7CaloL1L2L3'])
ic5CaloJetsL1L2L3 = ic5CaloJetsL1.clone(correctors = ['ic5CaloL1L2L3'])

ak5PFJetsL1L2L3 = ak5PFJetsL1.clone(correctors = ['ak5PFL1L2L3'])
ak7PFJetsL1L2L3 = ak7PFJetsL1.clone(correctors = ['ak7PFL1L2L3'])
kt4PFJetsL1L2L3 = kt4PFJetsL1.clone(correctors = ['kt4PFL1L2L3'])
kt6PFJetsL1L2L3 = kt6PFJetsL1.clone(correctors = ['kt6PFL1L2L3'])
sc5PFJetsL1L2L3 = sc5PFJetsL1.clone(correctors = ['sc5PFL1L2L3'])
sc7PFJetsL1L2L3 = sc7PFJetsL1.clone(correctors = ['sc7PFL1L2L3'])
ic5PFJetsL1L2L3 = ic5PFJetsL1.clone(correctors = ['ic5PFL1L2L3'])


# L2L3L6 CORRECTION PRODUCERS
ak5CaloJetsL2L3L6 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL2L3L6'])
ak7CaloJetsL2L3L6 = ak7CaloJetsL1.clone(correctors = ['ak7CaloL2L3L6'])
kt4CaloJetsL2L3L6 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL2L3L6'])
kt6CaloJetsL2L3L6 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL2L3L6'])
sc5CaloJetsL2L3L6 = sc5CaloJetsL1.clone(correctors = ['sc5CaloL2L3L6'])
sc7CaloJetsL2L3L6 = sc7CaloJetsL1.clone(correctors = ['sc7CaloL2L3L6'])
ic5CaloJetsL2L3L6 = ic5CaloJetsL1.clone(correctors = ['ic5CaloL2L3L6'])

ak5PFJetsL2L3L6 = ak5PFJetsL1.clone(correctors = ['ak5PFL2L3L6'])
ak7PFJetsL2L3L6 = ak7PFJetsL1.clone(correctors = ['ak7PFL2L3L6'])
kt4PFJetsL2L3L6 = kt4PFJetsL1.clone(correctors = ['kt4PFL2L3L6'])
kt6PFJetsL2L3L6 = kt6PFJetsL1.clone(correctors = ['kt6PFL2L3L6'])
sc5PFJetsL2L3L6 = sc5PFJetsL1.clone(correctors = ['sc5PFL2L3L6'])
sc7PFJetsL2L3L6 = sc7PFJetsL1.clone(correctors = ['sc7PFL2L3L6'])
ic5PFJetsL2L3L6 = ic5PFJetsL1.clone(correctors = ['ic5PFL2L3L6'])


# L1L2L3L6 CORRECTION PRODUCERS
ak5CaloJetsL1L2L3L6 = ak5CaloJetsL1.clone(correctors = ['ak5CaloL1L2L3L6'])
ak7CaloJetsL1L2L3L6 = ak7CaloJetsL1.clone(correctors = ['ak7CaloL1L2L3L6'])
kt4CaloJetsL1L2L3L6 = kt4CaloJetsL1.clone(correctors = ['kt4CaloL1L2L3L6'])
kt6CaloJetsL1L2L3L6 = kt6CaloJetsL1.clone(correctors = ['kt6CaloL1L2L3L6'])
sc5CaloJetsL1L2L3L6 = sc5CaloJetsL1.clone(correctors = ['sc5CaloL1L2L3L6'])
sc7CaloJetsL1L2L3L6 = sc7CaloJetsL1.clone(correctors = ['sc7CaloL1L2L3L6'])
ic5CaloJetsL1L2L3L6 = ic5CaloJetsL1.clone(correctors = ['ic5CaloL1L2L3L6'])

ak5PFJetsL1L2L3L6 = ak5PFJetsL1.clone(correctors = ['ak5PFL1L2L3L6'])
ak7PFJetsL1L2L3L6 = ak7PFJetsL1.clone(correctors = ['ak7PFL1L2L3L6'])
kt4PFJetsL1L2L3L6 = kt4PFJetsL1.clone(correctors = ['kt4PFL1L2L3L6'])
kt6PFJetsL1L2L3L6 = kt6PFJetsL1.clone(correctors = ['kt6PFL1L2L3L6'])
sc5PFJetsL1L2L3L6 = sc5PFJetsL1.clone(correctors = ['sc5PFL1L2L3L6'])
sc7PFJetsL1L2L3L6 = sc7PFJetsL1.clone(correctors = ['sc7PFL1L2L3L6'])
ic5PFJetsL1L2L3L6 = ic5PFJetsL1.clone(correctors = ['ic5PFL1L2L3L6'])
