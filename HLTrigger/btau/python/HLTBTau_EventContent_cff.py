# The following comments couldn't be translated into the new config version:

# Calorimeter

# Si Pixel hits

#Pixel Reco

# Si Strip hits

#CaloReco as input of L2 Tau

# Calorimeter

#Pixel Reco

#CaloReco as input of L2 Tau

import FWCore.ParameterSet.Config as cms

from HLTrigger.btau.jetTag.eventContent_Lifetime_cff import *
from HLTrigger.btau.jetTag.eventContent_SoftMuon_cff import *
from HLTrigger.btau.displacedmumu.eventContent_DisplacedMumu_cff import *
from HLTrigger.btau.displacedmumu.eventContent_Mumuk_cff import *
from HLTrigger.btau.tau.eventContent_SingleTau_cff import *
from HLTrigger.btau.tau.eventContent_SingleTauMET_cff import *
from HLTrigger.btau.tau.eventContent_DoubleTau_cff import *
from HLTrigger.btau.tau.eventContent_DoubleTauSiStrip_cff import *
# Full Event content for btau HLT paths
HLTBTauFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_caloTowersTau*_*_*', 'keep *_towerMakerForAll_*_*', 'keep *_towerMakerForTaus_*_*', 'keep *_siPixelRecHits_*_*', 'keep *_pixelVertices_*_*', 'keep *_pixelTracks_*_*', 'keep *_siStripRecHits_*_*', 'keep *_siStripMatchedRecHits_*_*', 'keep *_icone5Tau1*_*_*', 'keep *_icone5Tau2*_*_*', 'keep *_icone5Tau3*_*_*', 'keep *_icone5Tau4*_*_*')
)
# RECO content for btau HLT paths
HLTBTauRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_caloTowersTau*_*_*', 'keep *_towerMakerForAll_*_*', 'keep *_towerMakerForTaus_*_*', 'keep *_pixelVertices_*_*', 'keep *_pixelTracks_*_*', 'keep *_icone5Tau1*_*_*', 'keep *_icone5Tau2*_*_*', 'keep *_icone5Tau3*_*_*', 'keep *_icone5Tau4*_*_*')
)
# AOD content for btau HLT paths
HLTBTauAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("isolatedL3SingleTau"), cms.InputTag("isolatedL3SingleTauMET"), cms.InputTag("isolatedL25PixelTau"), cms.InputTag("hltBLifetimeL3Jets"), cms.InputTag("hltBSoftmuonL25Jets"), cms.InputTag("hltMuTracks"), cms.InputTag("hltMuTracks"), cms.InputTag("hltMumukAllConeTracks")),
    triggerFilters = cms.VInputTag(cms.InputTag("filterL3SingleTau"), cms.InputTag("filterL3SingleTauMET"), cms.InputTag("filterL25PixelTau"), cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("hltBSoftmuonL3filter"), cms.InputTag("hltBSoftmuonByDRL3filter"), cms.InputTag("displacedJpsitoMumuFilter"), cms.InputTag("hltmmkFilter")),
    outputCommands = cms.untracked.vstring()
)
HLTBTauFEVT.outputCommands.extend(JetTagLifetimeHLT.outputCommands)
HLTBTauFEVT.outputCommands.extend(JetTagSoftMuonHLT.outputCommands)
HLTBTauFEVT.outputCommands.extend(DisplacedMumuHLT.outputCommands)
HLTBTauFEVT.outputCommands.extend(MumukHLT.outputCommands)
HLTBTauFEVT.outputCommands.extend(SingleTauHLT.outputCommands)
HLTBTauFEVT.outputCommands.extend(SingleTauMETHLT.outputCommands)
HLTBTauFEVT.outputCommands.extend(DoubleTauHLT.outputCommands)
HLTBTauRECO.outputCommands.extend(JetTagLifetimeHLT.outputCommands)
HLTBTauRECO.outputCommands.extend(JetTagSoftMuonHLT.outputCommands)
HLTBTauRECO.outputCommands.extend(DisplacedMumuHLT.outputCommands)
HLTBTauRECO.outputCommands.extend(MumukHLT.outputCommands)
HLTBTauRECO.outputCommands.extend(SingleTauHLT.outputCommands)
HLTBTauRECO.outputCommands.extend(SingleTauMETHLT.outputCommands)
HLTBTauRECO.outputCommands.extend(DoubleTauHLT.outputCommands)

