import FWCore.ParameterSet.Config as cms

##------------------  IMPORT THE SERVICES  ----------------------
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
from JetMETCorrections.Configuration.JetCorrectionCondDB_cff import *
##------------------  DEFINE THE PRODUCER MODULES  --------------

##------------------  CALO JETS ---------------------------------
ak5CaloJetsL2L3 = cms.EDProducer('CaloJetCorrectionProducer',
     src        = cms.InputTag('ak5CaloJets'),
     correctors = cms.vstring('ak5CaloL2L3')
    )

ak7CaloJetsL2L3 = ak5CaloJetsL2L3.clone(src = 'ak7CaloJets', correctors = ['ak7CaloL2L3'])
sc5CaloJetsL2L3 = ak5CaloJetsL2L3.clone(src = 'sc5CaloJets', correctors = ['sc5CaloL2L3'])
sc7CaloJetsL2L3 = ak5CaloJetsL2L3.clone(src = 'sc7CaloJets', correctors = ['sc7CaloL2L3'])
kt4CaloJetsL2L3 = ak5CaloJetsL2L3.clone(src = 'kt4CaloJets', correctors = ['kt4CaloL2L3'])
kt6CaloJetsL2L3 = ak5CaloJetsL2L3.clone(src = 'kt6CaloJets', correctors = ['kt6CaloL2L3'])
ic5CaloJetsL2L3 = ak5CaloJetsL2L3.clone(src = 'ic5CaloJets', correctors = ['ic5CaloL2L3'])
##------------------  PF JETS -----------------------------------
ak5PFJetsL2L3   = cms.EDProducer('PFJetCorrectionProducer',
     src        = cms.InputTag('ak5PFJets'),
     correctors = cms.vstring('ak5PFL2L3')
    )

ak7PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL2L3'])
sc5PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'sc5PFJets', correctors = ['sc5PFL2L3'])
sc7PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'sc7PFJets', correctors = ['sc7PFL2L3'])
kt4PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL2L3'])
kt6PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL2L3'])
ic5PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'ic5PFJets', correctors = ['ic5PFL2L3'])
