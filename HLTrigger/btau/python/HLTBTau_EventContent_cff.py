# The following comments couldn't be translated into the new config version:

# Calorimeter

# Si Pixel hits

# Pixel Reco

# Si Strip hits

# CaloReco as input of L2 Tau

# Calorimeter

# Pixel Reco

# CaloReco as input of L2 Tau

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
    outputCommands = cms.untracked.vstring('keep *_hltCaloTowersTau*_*_*', 
        'keep *_hltTowerMakerForAll_*_*', 
        'keep *_hltTowerMakerForTaus_*_*', 
        'keep *_hltSiPixelRecHits_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltSiStripRecHits_*_*', 
        'keep *_hltSiStripMatchedRecHits_*_*', 
        'keep *_hltIcone5Tau1*_*_*', 
        'keep *_hltIcone5Tau2*_*_*', 
        'keep *_hltIcone5Tau3*_*_*', 
        'keep *_hltIcone5Tau4*_*_*')
)
# RECO content for btau HLT paths
HLTBTauRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltCaloTowersTau*_*_*', 
        'keep *_hltTowerMakerForAll_*_*', 
        'keep *_hltTowerMakerForTaus_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltIcone5Tau1*_*_*', 
        'keep *_hltIcone5Tau2*_*_*', 
        'keep *_hltIcone5Tau3*_*_*', 
        'keep *_hltIcone5Tau4*_*_*')
)
# AOD content for btau HLT paths
HLTBTauAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltIsolatedL3SingleTau"), cms.InputTag("hltIsolatedL3SingleTauMET"), cms.InputTag("hltIsolatedL25PixelTau"), cms.InputTag("hltIsolatedL3SingleTauRelaxed"), cms.InputTag("hltIsolatedL3SingleTauMETRelaxed"), 
        cms.InputTag("hltIsolatedL25PixelTauRelaxed"), cms.InputTag("hltBLifetimeL3Jets"), cms.InputTag("hltBSoftmuonL25Jets"), cms.InputTag("hltMuTracks"), cms.InputTag("hltMuTracks"), 
        cms.InputTag("hltMumukAllConeTracks")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltFilterL3SingleTau"), cms.InputTag("hltFilterL3SingleTauMET"), cms.InputTag("hltFilterL25PixelTau"), cms.InputTag("hltFilterL3SingleTauRelaxed"), cms.InputTag("hltFilterL3SingleTauMETRelaxed"), 
        cms.InputTag("hltFilterL25PixelTauRelaxed"), cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("hltBSoftmuonL3filter"), cms.InputTag("hltBSoftmuonByDRL3filter"), cms.InputTag("hltDisplacedJpsitoMumuFilter"), 
        cms.InputTag("hltDisplacedJpsitoMumuFilterRelaxed"), cms.InputTag("hltmmkFilter")),
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

