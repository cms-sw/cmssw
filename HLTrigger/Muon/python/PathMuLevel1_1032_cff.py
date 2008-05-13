# The following comments couldn't be translated into the new config version:

#L1 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_1032_cff import *
import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi
# HLT Filter flux ##########################################################
hltMuLevel1PathLevel1Seed = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
import HLTrigger.HLTcore.hltPrescaler_cfi
hltPrescalehltMuLevel1Path = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
hltMuLevel1PathL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltMuLevel1PathLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

muLevel1Path = cms.Sequence(hltPrescalehltMuLevel1Path+hltL1muonrecoSequence+hltMuLevel1PathLevel1Seed+hltMuLevel1PathL1Filtered)
hltMuLevel1PathLevel1Seed.L1SeedsLogicalExpression = 'L1_SingleMu3 OR L1_SingleMu5 OR L1_SingleMu7 OR L1_DoubleMu3'
hltPrescalehltMuLevel1Path.prescaleFactor = 1000

