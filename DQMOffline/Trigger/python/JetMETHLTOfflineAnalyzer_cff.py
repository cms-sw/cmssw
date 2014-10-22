import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.CorrectedJetProducers_cff import *

from DQMOffline.Trigger.JetMETHLTOfflineSource_cfi import *

jetMETHLTOfflineAnalyzer = cms.Sequence(
    ak4CaloL1FastL2L3CorrectorChain
    * ak4PFL1FastL2L3CorrectorChain
    * ak4CaloL1FastL2L3ResidualCorrectorChain
    * ak4PFL1FastL2L3ResidualCorrectorChain
    #* ak4CaloJetsL1FastL2L3
    #* ak4PFJetsL1FastL2L3
    #* ak4CaloJetsL1FastL2L3Residual
    #* ak4PFJetsL1FastL2L3Residual
    * jetMETHLTOfflineSource
)
