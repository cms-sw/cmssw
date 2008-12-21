import FWCore.ParameterSet.Config as cms

from DQM.HLTEvF.HLTMonJetMET_cfi import *
from DQM.HLTEvF.jetmetDQMConsumer_cfi import *
#hltMonJM = cms.Path(hltMonJetMET)
hltMonJM = cms.Path(hltMonJetMET*jetmetDQMConsumer)

