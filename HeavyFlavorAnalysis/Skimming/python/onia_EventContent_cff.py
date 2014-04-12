import FWCore.ParameterSet.Config as cms

from HeavyFlavorAnalysis.Skimming.jpsiToMuMu_EventContent_cff import *
from HeavyFlavorAnalysis.Skimming.upsilonToMuMu_EventContent_cff import *
from HeavyFlavorAnalysis.Skimming.bToMuMu_EventContent_cff import *
from Configuration.EventContent.EventContent_cff import *
oniaMuMuEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_source_*_*', 
        'keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep recoCandidatesOwned_allMuons_*_*', 
        'keep recoCandidatesOwned_allTracks_*_*', 
        'keep recoCandidatesOwned_allStandAloneMuonTracks_*_*')
)
AODSIMEventContent.outputCommands.extend(oniaMuMuEventContent.outputCommands)
AODSIMEventContent.outputCommands.extend(bToMuMuEventContent.outputCommands)

