import FWCore.ParameterSet.Config as cms

##------------------  IMPORT THE SERVICES  ----------------------
from JetMETCorrections.Configuration.JetCorrectionServices_cff import *

##------------------  DEFINE THE PRODUCER MODULES  --------------

##------------------  CALO JETS ---------------------------------
L2L3CorJetAK5Calo = cms.EDProducer('CaloJetCorrectionProducer',
    src        = cms.InputTag('ak5CaloJets'),
    correctors = cms.vstring('ak5CaloL2L3')
    )

L2L3CorJetAK7Calo = L2L3CorJetAK5Calo.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL2L3'])
L2L3CorJetSC5Calo = L2L3CorJetAK5Calo.clone(src = 'sc5CaloJets', correctors = ['sc5CaloL2L3'])
L2L3CorJetSC7Calo = L2L3CorJetAK5Calo.clone(src = 'sc7CaloJets', correctors = ['sc7CaloL2L3'])
L2L3CorJetKT4Calo = L2L3CorJetAK5Calo.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL2L3'])
L2L3CorJetKT6Calo = L2L3CorJetAK5Calo.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL2L3'])
L2L3CorJetIC5Calo = L2L3CorJetAK5Calo.clone(src = 'ic5CaloJets', correctors = ['ic5CaloL2L3'])
##------------------  PF JETS -----------------------------------
L2L3CorJetAK5PF   = cms.EDProducer('PFJetCorrectionProducer',
    src        = cms.InputTag('ak5PFJets'),
    correctors = cms.vstring('ak5PFL2L3')
    )

L2L3CorJetAK7PF   = L2L3CorJetAK5PF.clone(src = 'ak7PFJets', correctors = ['ak7PFL2L3'])
L2L3CorJetSC5PF   = L2L3CorJetAK5PF.clone(src = 'sc5PFJets', correctors = ['sc5PFL2L3'])
L2L3CorJetSC7PF   = L2L3CorJetAK5PF.clone(src = 'sc7PFJets', correctors = ['sc7PFL2L3'])
L2L3CorJetKT4PF   = L2L3CorJetAK5PF.clone(src = 'kt4PFJets', correctors = ['kt4PFL2L3'])
L2L3CorJetKT6PF   = L2L3CorJetAK5PF.clone(src = 'kt6PFJets', correctors = ['kt6PFL2L3'])
L2L3CorJetIC5PF   = L2L3CorJetAK5PF.clone(src = 'ic5PFJets', correctors = ['ic5PFL2L3'])
