import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectorsAllAlgos_cff import *
from DQMOffline.Trigger.JetMETHLTOfflineSource_cfi import *

jetMETHLTOfflineAnalyzer = cms.Sequence(
    ak4CaloL1FastL2L3ResidualCorrectorChain
    * ak4PFL1FastL2L3ResidualCorrectorChain
    * jetMETHLTOfflineSourceAK4
#    * ak8PFCHSL1FastjetL2L3ResidualCorrectorChain #not working in all matrix tests, yet
    * jetMETHLTOfflineSourceAK8
    * jetMETHLTOfflineSourceAK4Fwd
    * jetMETHLTOfflineSourceAK8Fwd
)

jmeHLTDQMSourceExtra = cms.Sequence(
)
