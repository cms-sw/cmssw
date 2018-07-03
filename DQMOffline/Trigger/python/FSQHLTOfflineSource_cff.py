import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.FSQHLTOfflineSource_cfi import *

from JetMETCorrections.Configuration.CorrectedJetProducers_cff import *
fsqHLTOfflineSourceSequence = cms.Sequence(ak4PFL1FastL2L3CorrectorChain + fsqHLTOfflineSource)

fsqHLTDQMSourceExtra = cms.Sequence(
)

