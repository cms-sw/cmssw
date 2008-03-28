# The following comments couldn't be translated into the new config version:

#L2 muon

#L2 muon isolation

#L3 muon

#L3 muon isolation

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_cff import *
recoHLTMuIso = cms.Sequence(l1muonreco+l2muonreco+l2muonisoreco+l3muonreco+l3muonisoreco)

