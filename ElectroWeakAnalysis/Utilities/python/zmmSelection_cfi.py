import FWCore.ParameterSet.Config as cms

import copy

# Trigger requirements
import HLTrigger.HLTfilters.hltHighLevel_cfi
dimuonsHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
dimuonsHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
dimuonsHLTFilter.HLTPaths = ["HLT_Mu9"]

# Cuts for each muon
goodAODGlobalMuons = cms.EDFilter("MuonViewRefSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('isGlobalMuon=1 & pt>20 & abs(eta)<2.1 & isolationR03().sumPt/pt<0.1'),
  filter = cms.bool(True)
)

# Cuts on dimuon system
dimuonsGlobalAOD = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string('mass>60 & charge=0'),
    decay = cms.string("goodAODGlobalMuons@+ goodAODGlobalMuons@-")
)
dimuonsFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuonsGlobalAOD"),
    minNumber = cms.uint32(1)
)

# Selection sequence
goldenZMMSelectionSequence = cms.Sequence(
   dimuonsHLTFilter
   * goodAODGlobalMuons
   * dimuonsGlobalAOD*dimuonsFilter
)
