import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionProducers_cff import *

ak5CaloJetsL1FastL2L3         = ak5CaloJetsL1.clone(correctors = ['ak5CaloL1FastL2L3'])
ak5PFJetsL1FastL2L3           = ak5PFJetsL1.clone(correctors   = ['ak5PFL1FastL2L3'])
ak5CaloJetsL1FastL2L3Residual = ak5CaloJetsL1.clone(correctors = ['ak5CaloL1FastL2L3Residual'])
ak5PFJetsL1FastL2L3Residual   = ak5PFJetsL1.clone(correctors   = ['ak5PFL1FastL2L3Residual'])

from DQMOffline.Trigger.JetMETHLTOfflineSource_cfi import *

jetMETHLTOfflineAnalyzer = cms.Sequence(
    ak5CaloJetsL1FastL2L3
    * ak5PFJetsL1FastL2L3
    * ak5CaloJetsL1FastL2L3Residual
    * ak5PFJetsL1FastL2L3Residual
    * jetMETHLTOfflineSource
)
