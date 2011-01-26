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
##------------------  PF JETS -----------------------------------
ak5PFJetsL2L3   = cms.EDProducer('PFJetCorrectionProducer',
    src         = cms.InputTag('ak5PFJets'),
    correctors  = cms.vstring('ak5PFL2L3')
    )

ak7PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'ak7PFJets', correctors = ['ak7PFL2L3'])
kt4PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'kt4PFJets', correctors = ['kt4PFL2L3'])
kt6PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'kt6PFJets', correctors = ['kt6PFL2L3'])
ic5PFJetsL2L3   = ak5PFJetsL2L3.clone(src = 'iterativeCone5PFJets', correctors = ['ic5PFL2L3'])
##------------------  JPT JETS ----------------------------------
#ak5JPTJetsL2L3   = cms.EDProducer('JPTJetCorrectionProducer',
#    src         = cms.InputTag('JetPlusTrackZSPCorJetAntiKt5'),
#    correctors  = cms.vstring('ak5JPTL2L3')
#    )
#ic5JPTJetsL2L3   = ak5JPTJetsL2L3.clone(src = 'ic5JPTJets', correctors = ['ic5JPTL2L3'])
##------------------  TRK JETS ----------------------------------
#ak5TrackJetsL2L3   = cms.EDProducer('TrackJetCorrectionProducer',
#    src         = cms.InputTag('ak5TrackJets'),
#    correctors  = cms.vstring('ak5TrackL2L3')
#    )
