import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.common.CaloTowers_cff import *
from HLTrigger.Configuration.common.RecoJetMET_cff import *
hltBCommonL2reco = cms.Sequence(doCalo+doHLTJetReco+doHLTHTReco)

