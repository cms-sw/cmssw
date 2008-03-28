import FWCore.ParameterSet.Config as cms

#
from HLTrigger.special.eventContent_HcalIsolatedTrack_cff import *
from HLTrigger.special.eventContent_AlcastreamEcalPi0_cff import *
from HLTrigger.special.eventContent_AlcastreamEcalPhiSym_cff import *
from HLTrigger.special.eventContent_AlcastreamHcalPhiSym_cff import *
# Full Event content for Special HLT paths
HLTSpecialFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
# RECO content for Special HLT paths
HLTSpecialRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
# AOD content for Special HLT paths
HLTSpecialAOD = cms.PSet(
    triggerCollections = cms.VInputTag(),
    triggerFilters = cms.VInputTag(),
    outputCommands = cms.untracked.vstring()
)
HLTSpecialFEVT.outputCommands.extend(HLTHcalIsolatedTrackFEVT.outputCommands)
HLTSpecialFEVT.outputCommands.extend(HLTAlcaRecoEcalPi0StreamFEVT.outputCommands)
HLTSpecialFEVT.outputCommands.extend(HLTAlcaRecoEcalPhiSymStreamFEVT.outputCommands)
HLTSpecialFEVT.outputCommands.extend(HLTAlcaRecoHcalPhiSymStreamFEVT.outputCommands)
HLTSpecialRECO.outputCommands.extend(HLTHcalIsolatedTrackRECO.outputCommands)
HLTSpecialRECO.outputCommands.extend(HLTAlcaRecoEcalPi0StreamRECO.outputCommands)
HLTSpecialRECO.outputCommands.extend(HLTAlcaRecoEcalPhiSymStreamRECO.outputCommands)
HLTSpecialRECO.outputCommands.extend(HLTAlcaRecoHcalPhiSymStreamRECO.outputCommands)
HLTSpecialAOD.outputCommands.extend(HLTHcalIsolatedTrackAOD.outputCommands)
HLTSpecialAOD.outputCommands.extend(HLTAlcaRecoEcalPi0StreamAOD.outputCommands)
HLTSpecialAOD.outputCommands.extend(HLTAlcaRecoEcalPhiSymStreamAOD.outputCommands)
HLTSpecialAOD.outputCommands.extend(HLTAlcaRecoHcalPhiSymStreamAOD.outputCommands)
HLTSpecialAOD.triggerCollections.extend(HLTHcalIsolatedTrackAOD.triggerCollections)
HLTSpecialAOD.triggerFilters.extend(HLTHcalIsolatedTrackAOD.triggerFilters)
HLTSpecialAOD.triggerCollections.extend(HLTAlcaRecoEcalPi0StreamAOD.triggerCollections)
HLTSpecialAOD.triggerFilters.extend(HLTAlcaRecoEcalPi0StreamAOD.triggerFilters)
HLTSpecialAOD.triggerCollections.extend(HLTAlcaRecoEcalPhiSymStreamAOD.triggerCollections)
HLTSpecialAOD.triggerFilters.extend(HLTAlcaRecoEcalPhiSymStreamAOD.triggerFilters)
HLTSpecialAOD.triggerCollections.extend(HLTAlcaRecoHcalPhiSymStreamAOD.triggerCollections)
HLTSpecialAOD.triggerFilters.extend(HLTAlcaRecoHcalPhiSymStreamAOD.triggerFilters)

