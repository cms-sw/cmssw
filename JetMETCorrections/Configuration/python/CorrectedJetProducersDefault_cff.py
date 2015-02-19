import FWCore.ParameterSet.Config as cms

##------------------  IMPORT THE SERVICES  ----------------------
from JetMETCorrections.Configuration.JetCorrectorsAllAlgos_cff import *

##------------------  DEFINE THE PRODUCER MODULES  --------------

##------------------  CALO JETS ---------------------------------
ak4CaloJetsL2L3 = cms.EDProducer('CorrectedCaloJetProducer',
    src         = cms.InputTag('ak4CaloJets'),
    correctors  = cms.VInputTag('ak4CaloL2L3Corrector')
    )

ak7CaloJetsL2L3 = ak4CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL2L3Corrector'])
kt4CaloJetsL2L3 = ak4CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL2L3Corrector'])
kt6CaloJetsL2L3 = ak4CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL2L3Corrector'])
ic5CaloJetsL2L3 = ak4CaloJetsL2L3.clone(src = 'iterativeCone5CaloJets', correctors = ['ic5CaloL2L3Corrector'])

ak4CaloJetsL2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL2L3ResidualCorrector'])
ak7CaloJetsL2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL2L3ResidualCorrector'])
kt4CaloJetsL2L3Residual = ak4CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL2L3ResidualCorrector'])
kt6CaloJetsL2L3Residual = ak4CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL2L3ResidualCorrector'])
ic5CaloJetsL2L3Residual = ak4CaloJetsL2L3.clone(src = 'iterativeCone5CaloJets', correctors = ['ic5CaloL2L3ResidualCorrector'])

ak4CaloJetsL1L2L3 = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL1L2L3Corrector'])
ak7CaloJetsL1L2L3 = ak4CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL1L2L3Corrector'])
kt4CaloJetsL1L2L3 = ak4CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL1L2L3Corrector'])
kt6CaloJetsL1L2L3 = ak4CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL1L2L3Corrector'])
ic5CaloJetsL1L2L3 = ak4CaloJetsL2L3.clone(src = 'iterativeCone5CaloJets', correctors = ['ic5CaloL1L2L3Corrector'])

ak4CaloJetsL1FastL2L3 = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL1FastL2L3Corrector'])
ak7CaloJetsL1FastL2L3 = ak4CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL1FastL2L3Corrector'])
kt4CaloJetsL1FastL2L3 = ak4CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL1FastL2L3Corrector'])
kt6CaloJetsL1FastL2L3 = ak4CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL1FastL2L3Corrector'])
ic5CaloJetsL1FastL2L3 = ak4CaloJetsL2L3.clone(src = 'iterativeCone5CaloJets', correctors = ['ic5CaloL1FastL2L3Corrector'])

ak4CaloJetsL1L2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL1L2L3ResidualCorrector'])
ak7CaloJetsL1L2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL1L2L3ResidualCorrector'])
kt4CaloJetsL1L2L3Residual = ak4CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL1L2L3ResidualCorrector'])
kt6CaloJetsL1L2L3Residual = ak4CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL1L2L3ResidualCorrector'])
ic5CaloJetsL1L2L3Residual = ak4CaloJetsL2L3.clone(src = 'iterativeCone5CaloJets', correctors = ['ic5CaloL1L2L3ResidualCorrector'])

ak4CaloJetsL1FastL2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL1FastL2L3ResidualCorrector'])
ak7CaloJetsL1FastL2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL1FastL2L3ResidualCorrector'])
kt4CaloJetsL1FastL2L3Residual = ak4CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL1FastL2L3ResidualCorrector'])
kt6CaloJetsL1FastL2L3Residual = ak4CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL1FastL2L3ResidualCorrector'])
ic5CaloJetsL1FastL2L3Residual = ak4CaloJetsL2L3.clone(src = 'iterativeCone5CaloJets', correctors = ['ic5CaloL1FastL2L3ResidualCorrector'])

##------------------  PF JETS -----------------------------------
ak4PFJetsL2L3   = cms.EDProducer('PFJetCorrectionProducer',
    src         = cms.InputTag('ak4PFJets'),
    correctors  = cms.VInputTag('ak4PFL2L3Corrector')
    )
ak1PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'ak1PFJets', correctors = ['ak1PFL2L3Corrector'])
ak2PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'ak2PFJets', correctors = ['ak2PFL2L3Corrector'])
ak3PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'ak3PFJets', correctors = ['ak3PFL2L3Corrector'])
ak5PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'ak5PFJets', correctors = ['ak5PFL2L3Corrector'])
ak6PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'ak6PFJets', correctors = ['ak6PFL2L3Corrector'])
ak7PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL2L3Corrector'])
ak8PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'ak8PFJets', correctors = ['ak8PFL2L3Corrector'])
ak9PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'ak9PFJets', correctors = ['ak9PFL2L3Corrector'])
ak10PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'ak10PFJets', correctors = ['ak10PFL2L3Corrector'])
kt4PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL2L3Corrector'])
kt6PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL2L3Corrector'])
ic5PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL2L3Corrector'])

ak1PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak1PFJets', correctors = ['ak1PFL2L3ResidualCorrector'])
ak2PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak2PFJets', correctors = ['ak2PFL2L3ResidualCorrector'])
ak3PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak3PFJets', correctors = ['ak3PFL2L3ResidualCorrector'])
ak4PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL2L3ResidualCorrector'])
ak5PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak5PFJets', correctors = ['ak5PFL2L3ResidualCorrector'])
ak6PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak6PFJets', correctors = ['ak6PFL2L3ResidualCorrector'])
ak7PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL2L3ResidualCorrector'])
ak8PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak8PFJets', correctors = ['ak8PFL2L3ResidualCorrector'])
ak9PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak9PFJets', correctors = ['ak9PFL2L3ResidualCorrector'])
ak10PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak10PFJets', correctors = ['ak10PFL2L3ResidualCorrector'])
kt4PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL2L3ResidualCorrector'])
kt6PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL2L3ResidualCorrector'])
ic5PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL2L3ResidualCorrector'])

ak1PFJetsL1L2L3   = ak4PFJetsL2L3.clone(src = 'ak1PFJets', correctors = ['ak1PFL1L2L3Corrector'])
ak2PFJetsL1L2L3   = ak4PFJetsL2L3.clone(src = 'ak2PFJets', correctors = ['ak2PFL1L2L3Corrector'])
ak3PFJetsL1L2L3   = ak4PFJetsL2L3.clone(src = 'ak3PFJets', correctors = ['ak3PFL1L2L3Corrector'])
ak4PFJetsL1L2L3   = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL1L2L3Corrector'])
ak5PFJetsL1L2L3   = ak4PFJetsL2L3.clone(src = 'ak5PFJets', correctors = ['ak5PFL1L2L3Corrector'])
ak6PFJetsL1L2L3   = ak4PFJetsL2L3.clone(src = 'ak6PFJets', correctors = ['ak6PFL1L2L3Corrector'])
ak7PFJetsL1L2L3   = ak4PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL1L2L3Corrector'])
ak8PFJetsL1L2L3   = ak4PFJetsL2L3.clone(src = 'ak8PFJets', correctors = ['ak8PFL1L2L3Corrector'])
ak9PFJetsL1L2L3   = ak4PFJetsL2L3.clone(src = 'ak9PFJets', correctors = ['ak9PFL1L2L3Corrector'])
ak10PFJetsL1L2L3   = ak4PFJetsL2L3.clone(src = 'ak10PFJets', correctors = ['ak10PFL1L2L3Corrector'])
kt4PFJetsL1L2L3 = ak4PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL1L2L3Corrector'])
kt6PFJetsL1L2L3 = ak4PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL1L2L3Corrector'])
ic5PFJetsL1L2L3 = ak4PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL1L2L3Corrector'])

ak1PFJetsL1FastL2L3   = ak4PFJetsL2L3.clone(src = 'ak1PFJets', correctors = ['ak1PFL1FastL2L3Corrector'])
ak2PFJetsL1FastL2L3   = ak4PFJetsL2L3.clone(src = 'ak2PFJets', correctors = ['ak2PFL1FastL2L3Corrector'])
ak3PFJetsL1FastL2L3   = ak4PFJetsL2L3.clone(src = 'ak3PFJets', correctors = ['ak3PFL1FastL2L3Corrector'])
ak4PFJetsL1FastL2L3   = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL1FastL2L3Corrector'])
ak5PFJetsL1FastL2L3   = ak4PFJetsL2L3.clone(src = 'ak5PFJets', correctors = ['ak5PFL1FastL2L3Corrector'])
ak6PFJetsL1FastL2L3   = ak4PFJetsL2L3.clone(src = 'ak6PFJets', correctors = ['ak6PFL1FastL2L3Corrector'])
ak7PFJetsL1FastL2L3   = ak4PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL1FastL2L3Corrector'])
ak8PFJetsL1FastL2L3   = ak4PFJetsL2L3.clone(src = 'ak8PFJets', correctors = ['ak8PFL1FastL2L3Corrector'])
ak9PFJetsL1FastL2L3   = ak4PFJetsL2L3.clone(src = 'ak9PFJets', correctors = ['ak9PFL1FastL2L3Corrector'])
ak10PFJetsL1FastL2L3   = ak4PFJetsL2L3.clone(src = 'ak10PFJets', correctors = ['ak10PFL1FastL2L3Corrector'])

kt4PFJetsL1FastL2L3 = ak4PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL1FastL2L3Corrector'])
kt6PFJetsL1FastL2L3 = ak4PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL1FastL2L3Corrector'])
ic5PFJetsL1FastL2L3 = ak4PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL1FastL2L3Corrector'])

ak1PFJetsL1L2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak1PFJets', correctors = ['ak1PFL1L2L3ResidualCorrector'])
ak2PFJetsL1L2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak2PFJets', correctors = ['ak2PFL1L2L3ResidualCorrector'])
ak3PFJetsL1L2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak3PFJets', correctors = ['ak3PFL1L2L3ResidualCorrector'])
ak4PFJetsL1L2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL1L2L3ResidualCorrector'])
ak5PFJetsL1L2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak5PFJets', correctors = ['ak5PFL1L2L3ResidualCorrector'])
ak6PFJetsL1L2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak6PFJets', correctors = ['ak6PFL1L2L3ResidualCorrector'])
ak7PFJetsL1L2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL1L2L3ResidualCorrector'])
ak8PFJetsL1L2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak8PFJets', correctors = ['ak8PFL1L2L3ResidualCorrector'])
ak9PFJetsL1L2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak9PFJets', correctors = ['ak9PFL1L2L3ResidualCorrector'])
ak10PFJetsL1L2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak10PFJets', correctors = ['ak10PFL1L2L3ResidualCorrector'])

kt4PFJetsL1L2L3Residual = ak4PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL1L2L3ResidualCorrector'])
kt6PFJetsL1L2L3Residual = ak4PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL1L2L3ResidualCorrector'])
ic5PFJetsL1L2L3Residual = ak4PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL1L2L3ResidualCorrector'])

ak1PFJetsL1FastL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak1PFJets', correctors = ['ak1PFL1FastL2L3ResidualCorrector'])
ak2PFJetsL1FastL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak2PFJets', correctors = ['ak2PFL1FastL2L3ResidualCorrector'])
ak3PFJetsL1FastL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak3PFJets', correctors = ['ak3PFL1FastL2L3ResidualCorrector'])
ak4PFJetsL1FastL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL1FastL2L3ResidualCorrector'])
ak5PFJetsL1FastL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak5PFJets', correctors = ['ak5PFL1FastL2L3ResidualCorrector'])
ak6PFJetsL1FastL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak6PFJets', correctors = ['ak6PFL1FastL2L3ResidualCorrector'])
ak7PFJetsL1FastL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL1FastL2L3ResidualCorrector'])
ak8PFJetsL1FastL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak8PFJets', correctors = ['ak8PFL1FastL2L3ResidualCorrector'])
ak9PFJetsL1FastL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak9PFJets', correctors = ['ak9PFL1FastL2L3ResidualCorrector'])
ak10PFJetsL1FastL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak10PFJets', correctors = ['ak10PFL1FastL2L3ResidualCorrector'])

kt4PFJetsL1FastL2L3Residual = ak4PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL1FastL2L3ResidualCorrector'])
kt6PFJetsL1FastL2L3Residual = ak4PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL1FastL2L3ResidualCorrector'])
ic5PFJetsL1FastL2L3Residual = ak4PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL1FastL2L3ResidualCorrector'])

##------------------  JPT JETS ----------------------------------
ak4JPTJetsL2L3   = cms.EDProducer('JPTJetCorrectionProducer',
    src         = cms.InputTag('JetPlusTrackZSPCorJetAntiKt4'),
    correctors  = cms.VInputTag('ak4JPTL2L3Corrector')
    )

ak4JPTJetsL1L2L3 = ak4JPTJetsL2L3.clone(correctors = ['ak4JPTL1L2L3Corrector'])
ak4JPTJetsL1FastL2L3 = ak4JPTJetsL2L3.clone(correctors = ['ak4JPTL1FastL2L3Corrector'])
ak4JPTJetsL2L3Residual = ak4JPTJetsL2L3.clone(correctors = ['ak4JPTL2L3ResidualCorrector'])
ak4JPTJetsL1L2L3Residual = ak4JPTJetsL2L3.clone(correctors = ['ak4JPTL1L2L3ResidualCorrector'])
ak4JPTJetsL1FastL2L3Residual = ak4JPTJetsL2L3.clone(correctors = ['ak4JPTL1FastL2L3ResidualCorrector'])

##------------------  TRK JETS ----------------------------------
ak4TrackJetsL2L3   = cms.EDProducer('TrackJetCorrectionProducer',
    src         = cms.InputTag('ak4TrackJets'),
    correctors  = cms.VInputTag('ak4TrackL2L3Corrector')
    )
