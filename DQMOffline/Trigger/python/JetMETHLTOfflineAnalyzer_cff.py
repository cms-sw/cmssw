import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionProducers_cff import *

#ak4CaloJetsL1FastL2L3         = ak4CaloJetsL1.clone(correctors = ['ak4CaloL1FastL2L3'])
#ak4PFJetsL1FastL2L3           = ak4PFJetsL1.clone(correctors   = ['ak4PFL1FastL2L3'])
#ak4CaloJetsL1FastL2L3Residual = ak4CaloJetsL1.clone(correctors = ['ak4CaloL1FastL2L3Residual'])
#ak4PFJetsL1FastL2L3Residual   = ak4PFJetsL1.clone(correctors   = ['ak4PFL1FastL2L3Residual'])

from DQMOffline.Trigger.JetMETHLTOfflineSource_cfi import *

jetMETHLTOfflineAnalyzer = cms.Sequence(
    #ak4CaloJetsL1FastL2L3
    #* ak4PFJetsL1FastL2L3
    #* ak4CaloJetsL1FastL2L3Residual
    #* ak4PFJetsL1FastL2L3Residual
    jetMETHLTOfflineSource
)
