import FWCore.ParameterSet.Config as cms

from DQM.HLTEvF.HLTMonMuonDQM_cfi import *
from DQM.HLTEvF.HLTMonMuonBits_cfi import *
hltMonMuonDQM = cms.Path(hltMonMuDQM*hltMonMuBits)

