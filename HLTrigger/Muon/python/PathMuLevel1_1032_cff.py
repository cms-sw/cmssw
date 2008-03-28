# The following comments couldn't be translated into the new config version:

#L1 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_1032_cff import *
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
# HLT Filter flux ##########################################################
MuLevel1PathLevel1Seed = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescaleMuLevel1Path = copy.deepcopy(hltPrescaler)
MuLevel1PathL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("MuLevel1PathLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

muLevel1Path = cms.Sequence(prescaleMuLevel1Path+l1muonreco+MuLevel1PathLevel1Seed+MuLevel1PathL1Filtered)
MuLevel1PathLevel1Seed.L1SeedsLogicalExpression = 'L1_SingleMu3 OR L1_SingleMu5 OR L1_SingleMu7 OR L1_DoubleMu3'
prescaleMuLevel1Path.prescaleFactor = 1000

