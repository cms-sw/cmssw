import FWCore.ParameterSet.Config as cms

##------------------  IMPORT THE SERVICES  ----------------------
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *

##------------------  DEFINE THE PRODUCER MODULES  --------------

##------------------  CALO JETS ---------------------------------
ak4CaloJetsL2L3 = cms.EDProducer('CaloJetCorrectionProducer',
    src         = cms.InputTag('ak4CaloJets'),
    correctors  = cms.vstring('ak4CaloL2L3')
    )

ak8CaloJetsL2L3 = ak4CaloJetsL2L3.clone(src = 'ak8CaloJets', correctors = ['ak8CaloL2L3'])
kt4CaloJetsL2L3 = ak4CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL2L3'])
kt6CaloJetsL2L3 = ak4CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL2L3'])
ak4CaloJetsL2L3 = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL2L3'])

ak4CaloJetsL2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL2L3Residual'])
ak8CaloJetsL2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak8CaloJets', correctors = ['ak8CaloL2L3Residual'])
kt4CaloJetsL2L3Residual = ak4CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL2L3Residual'])
kt6CaloJetsL2L3Residual = ak4CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL2L3Residual'])
ak4CaloJetsL2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL2L3Residual'])

ak4CaloJetsL1L2L3 = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL1L2L3'])
ak8CaloJetsL1L2L3 = ak4CaloJetsL2L3.clone(src = 'ak8CaloJets', correctors = ['ak8CaloL1L2L3'])
kt4CaloJetsL1L2L3 = ak4CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL1L2L3'])
kt6CaloJetsL1L2L3 = ak4CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL1L2L3'])
ak4CaloJetsL1L2L3 = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL1L2L3'])

ak4CaloJetsL1FastL2L3 = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL1FastL2L3'])
ak8CaloJetsL1FastL2L3 = ak4CaloJetsL2L3.clone(src = 'ak8CaloJets', correctors = ['ak8CaloL1FastL2L3'])
kt4CaloJetsL1FastL2L3 = ak4CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL1FastL2L3'])
kt6CaloJetsL1FastL2L3 = ak4CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL1FastL2L3'])
ak4CaloJetsL1FastL2L3 = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL1FastL2L3'])

ak4CaloJetsL1L2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL1L2L3Residual'])
ak8CaloJetsL1L2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak8CaloJets', correctors = ['ak8CaloL1L2L3Residual'])
kt4CaloJetsL1L2L3Residual = ak4CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL1L2L3Residual'])
kt6CaloJetsL1L2L3Residual = ak4CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL1L2L3Residual'])
ak4CaloJetsL1L2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL1L2L3Residual'])

ak4CaloJetsL1FastL2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL1FastL2L3Residual'])
ak8CaloJetsL1FastL2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak8CaloJets', correctors = ['ak8CaloL1FastL2L3Residual'])
kt4CaloJetsL1FastL2L3Residual = ak4CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL1FastL2L3Residual'])
kt6CaloJetsL1FastL2L3Residual = ak4CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL1FastL2L3Residual'])
ak4CaloJetsL1FastL2L3Residual = ak4CaloJetsL2L3.clone(src = 'ak4CaloJets', correctors = ['ak4CaloL1FastL2L3Residual'])

##------------------  PF JETS -----------------------------------
ak4PFJetsL2L3   = cms.EDProducer('PFJetCorrectionProducer',
    src         = cms.InputTag('ak4PFJets'),
    correctors  = cms.vstring('ak4PFL2L3')
    )

ak8PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'ak8PFJets', correctors = ['ak8PFL2L3'])
kt4PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL2L3'])
kt6PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL2L3'])
ak4PFJetsL2L3   = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL2L3'])

ak4PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL2L3Residual'])
ak8PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak8PFJets', correctors = ['ak8PFL2L3Residual'])
kt4PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL2L3Residual'])
kt6PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL2L3Residual'])
ak4PFJetsL2L3Residual   = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL2L3Residual'])

ak4PFJetsL1L2L3 = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL1L2L3'])
ak8PFJetsL1L2L3 = ak4PFJetsL2L3.clone(src = 'ak8PFJets', correctors = ['ak8PFL1L2L3'])
kt4PFJetsL1L2L3 = ak4PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL1L2L3'])
kt6PFJetsL1L2L3 = ak4PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL1L2L3'])
ak4PFJetsL1L2L3 = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL1L2L3'])

ak4PFJetsL1FastL2L3 = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL1FastL2L3'])
ak8PFJetsL1FastL2L3 = ak4PFJetsL2L3.clone(src = 'ak8PFJets', correctors = ['ak8PFL1FastL2L3'])
kt4PFJetsL1FastL2L3 = ak4PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL1FastL2L3'])
kt6PFJetsL1FastL2L3 = ak4PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL1FastL2L3'])
ak4PFJetsL1FastL2L3 = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL1FastL2L3'])

ak4PFJetsL1L2L3Residual = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL1L2L3Residual'])
ak8PFJetsL1L2L3Residual = ak4PFJetsL2L3.clone(src = 'ak8PFJets', correctors = ['ak8PFL1L2L3Residual'])
kt4PFJetsL1L2L3Residual = ak4PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL1L2L3Residual'])
kt6PFJetsL1L2L3Residual = ak4PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL1L2L3Residual'])
ak4PFJetsL1L2L3Residual = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL1L2L3Residual'])

ak4PFJetsL1FastL2L3Residual = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL1FastL2L3Residual'])
ak8PFJetsL1FastL2L3Residual = ak4PFJetsL2L3.clone(src = 'ak8PFJets', correctors = ['ak8PFL1FastL2L3Residual'])
kt4PFJetsL1FastL2L3Residual = ak4PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL1FastL2L3Residual'])
kt6PFJetsL1FastL2L3Residual = ak4PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL1FastL2L3Residual'])
ak4PFJetsL1FastL2L3Residual = ak4PFJetsL2L3.clone(src = 'ak4PFJets', correctors = ['ak4PFL1FastL2L3Residual'])

##------------------  JPT JETS ----------------------------------
ak4JPTJetsL2L3   = cms.EDProducer('JPTJetCorrectionProducer',
    src         = cms.InputTag('JetPlusTrackZSPCorJetAntiKt5'),
    correctors  = cms.vstring('ak4JPTL2L3')
    )

ak4JPTJetsL1L2L3 = ak4JPTJetsL2L3.clone(correctors = ['ak4JPTL1L2L3'])
ak4JPTJetsL1FastL2L3 = ak4JPTJetsL2L3.clone(correctors = ['ak4JPTL1FastL2L3'])
ak4JPTJetsL2L3Residual = ak4JPTJetsL2L3.clone(correctors = ['ak4JPTL2L3Residual'])
ak4JPTJetsL1L2L3Residual = ak4JPTJetsL2L3.clone(correctors = ['ak4JPTL1L2L3Residual'])
ak4JPTJetsL1FastL2L3Residual = ak4JPTJetsL2L3.clone(correctors = ['ak4JPTL1FastL2L3Residual'])

##------------------  TRK JETS ----------------------------------
ak4TrackJetsL2L3   = cms.EDProducer('TrackJetCorrectionProducer',
    src         = cms.InputTag('ak4TrackJets'),
    correctors  = cms.vstring('ak4TrackL2L3')
    )
