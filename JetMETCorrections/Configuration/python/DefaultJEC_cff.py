import FWCore.ParameterSet.Config as cms

##------------------  IMPORT THE SERVICES  ----------------------
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *

##------------------  DEFINE THE PRODUCER MODULES  --------------

##------------------  CALO JETS ---------------------------------
ak5CaloJetsL2L3 = cms.EDProducer('CaloJetCorrectionProducer',
    src         = cms.InputTag('ak5CaloJets'),
    correctors  = cms.vstring('ak5CaloL2L3')
    )

ak7CaloJetsL2L3 = ak5CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL2L3'])
kt4CaloJetsL2L3 = ak5CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL2L3'])
kt6CaloJetsL2L3 = ak5CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL2L3'])
ic5CaloJetsL2L3 = ak5CaloJetsL2L3.clone(src = 'iterativeCone5CaloJets', correctors = ['ic5CaloL2L3'])

ak5CaloJetsL2L3Residual = ak5CaloJetsL2L3.clone(src = 'ak5CaloJets', correctors = ['ak5CaloL2L3Residual'])
ak7CaloJetsL2L3Residual = ak5CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL2L3Residual'])
kt4CaloJetsL2L3Residual = ak5CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL2L3Residual'])
kt6CaloJetsL2L3Residual = ak5CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL2L3Residual'])
ic5CaloJetsL2L3Residual = ak5CaloJetsL2L3.clone(src = 'iterativeCone5CaloJets', correctors = ['ic5CaloL2L3Residual'])

ak5CaloJetsL1L2L3 = ak5CaloJetsL2L3.clone(src = 'ak5CaloJets', correctors = ['ak5CaloL1L2L3'])
ak7CaloJetsL1L2L3 = ak5CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL1L2L3'])
kt4CaloJetsL1L2L3 = ak5CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL1L2L3'])
kt6CaloJetsL1L2L3 = ak5CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL1L2L3'])
ic5CaloJetsL1L2L3 = ak5CaloJetsL2L3.clone(src = 'iterativeCone5CaloJets', correctors = ['ic5CaloL1L2L3'])

ak5CaloJetsL1FastL2L3 = ak5CaloJetsL2L3.clone(src = 'ak5CaloJets', correctors = ['ak5CaloL1FastL2L3'])
ak7CaloJetsL1FastL2L3 = ak5CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL1FastL2L3'])
kt4CaloJetsL1FastL2L3 = ak5CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL1FastL2L3'])
kt6CaloJetsL1FastL2L3 = ak5CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL1FastL2L3'])
ic5CaloJetsL1FastL2L3 = ak5CaloJetsL2L3.clone(src = 'iterativeCone5CaloJets', correctors = ['ic5CaloL1FastL2L3'])

ak5CaloJetsL1L2L3Residual = ak5CaloJetsL2L3.clone(src = 'ak5CaloJets', correctors = ['ak5CaloL1L2L3Residual'])
ak7CaloJetsL1L2L3Residual = ak5CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL1L2L3Residual'])
kt4CaloJetsL1L2L3Residual = ak5CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL1L2L3Residual'])
kt6CaloJetsL1L2L3Residual = ak5CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL1L2L3Residual'])
ic5CaloJetsL1L2L3Residual = ak5CaloJetsL2L3.clone(src = 'iterativeCone5CaloJets', correctors = ['ic5CaloL1L2L3Residual'])

ak5CaloJetsL1FastL2L3Residual = ak5CaloJetsL2L3.clone(src = 'ak5CaloJets', correctors = ['ak5CaloL1FastL2L3Residual'])
ak7CaloJetsL1FastL2L3Residual = ak5CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL1FastL2L3Residual'])
kt4CaloJetsL1FastL2L3Residual = ak5CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL1FastL2L3Residual'])
kt6CaloJetsL1FastL2L3Residual = ak5CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL1FastL2L3Residual'])
ic5CaloJetsL1FastL2L3Residual = ak5CaloJetsL2L3.clone(src = 'iterativeCone5CaloJets', correctors = ['ic5CaloL1FastL2L3Residual'])

##------------------  PF JETS -----------------------------------
ak5PFJetsL2L3   = cms.EDProducer('PFJetCorrectionProducer',
    src         = cms.InputTag('ak5PFJets'),
    correctors  = cms.vstring('ak5PFL2L3')
    )

ak7PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL2L3'])
kt4PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL2L3'])
kt6PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL2L3'])
ic5PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL2L3'])

ak5PFJetsL2L3Residual   = ak5PFJetsL2L3.clone(src = 'ak5PFJets', correctors = ['ak5PFL2L3Residual'])
ak7PFJetsL2L3Residual   = ak5PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL2L3Residual'])
kt4PFJetsL2L3Residual   = ak5PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL2L3Residual'])
kt6PFJetsL2L3Residual   = ak5PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL2L3Residual'])
ic5PFJetsL2L3Residual   = ak5PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL2L3Residual'])

ak5PFJetsL1L2L3 = ak5PFJetsL2L3.clone(src = 'ak5PFJets', correctors = ['ak5PFL1L2L3'])
ak7PFJetsL1L2L3 = ak5PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL1L2L3'])
kt4PFJetsL1L2L3 = ak5PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL1L2L3'])
kt6PFJetsL1L2L3 = ak5PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL1L2L3'])
ic5PFJetsL1L2L3 = ak5PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL1L2L3'])

ak5PFJetsL1FastL2L3 = ak5PFJetsL2L3.clone(src = 'ak5PFJets', correctors = ['ak5PFL1FastL2L3'])
ak7PFJetsL1FastL2L3 = ak5PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL1FastL2L3'])
kt4PFJetsL1FastL2L3 = ak5PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL1FastL2L3'])
kt6PFJetsL1FastL2L3 = ak5PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL1FastL2L3'])
ic5PFJetsL1FastL2L3 = ak5PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL1FastL2L3'])

ak5PFJetsL1L2L3Residual = ak5PFJetsL2L3.clone(src = 'ak5PFJets', correctors = ['ak5PFL1L2L3Residual'])
ak7PFJetsL1L2L3Residual = ak5PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL1L2L3Residual'])
kt4PFJetsL1L2L3Residual = ak5PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL1L2L3Residual'])
kt6PFJetsL1L2L3Residual = ak5PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL1L2L3Residual'])
ic5PFJetsL1L2L3Residual = ak5PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL1L2L3Residual'])

ak5PFJetsL1FastL2L3Residual = ak5PFJetsL2L3.clone(src = 'ak5PFJets', correctors = ['ak5PFL1FastL2L3Residual'])
ak7PFJetsL1FastL2L3Residual = ak5PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL1FastL2L3Residual'])
kt4PFJetsL1FastL2L3Residual = ak5PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL1FastL2L3Residual'])
kt6PFJetsL1FastL2L3Residual = ak5PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL1FastL2L3Residual'])
ic5PFJetsL1FastL2L3Residual = ak5PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL1FastL2L3Residual'])

##------------------  JPT JETS ----------------------------------
ak5JPTJetsL2L3   = cms.EDProducer('JPTJetCorrectionProducer',
    src         = cms.InputTag('JetPlusTrackZSPCorJetAntiKt5'),
    correctors  = cms.vstring('ak5JPTL2L3')
    )

ak5JPTJetsL1L2L3 = ak5JPTJetsL2L3.clone(correctors = ['ak5JPTL1L2L3'])
ak5JPTJetsL1FastL2L3 = ak5JPTJetsL2L3.clone(correctors = ['ak5JPTL1FastL2L3'])
ak5JPTJetsL2L3Residual = ak5JPTJetsL2L3.clone(correctors = ['ak5JPTL2L3Residual'])
ak5JPTJetsL1L2L3Residual = ak5JPTJetsL2L3.clone(correctors = ['ak5JPTL1L2L3Residual'])
ak5JPTJetsL1FastL2L3Residual = ak5JPTJetsL2L3.clone(correctors = ['ak5JPTL1FastL2L3Residual'])

##------------------  TRK JETS ----------------------------------
ak5TrackJetsL2L3   = cms.EDProducer('TrackJetCorrectionProducer',
    src         = cms.InputTag('ak5TrackJets'),
    correctors  = cms.vstring('ak5TrackL2L3')
    )
