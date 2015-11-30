# AlCaReco for track based alignment using isolated muon tracks
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlMuonIsolated_cff import *

ALCARECOTkAlMuonIsolatedHIHLT = ALCARECOTkAlMuonIsolatedHLT.clone(
    eventSetupPathsKey = 'TkAlMuonIsolated'
    )

ALCARECOTkAlMuonIsolatedHIDCSFilter = ALCARECOTkAlMuonIsolatedDCSFilter.clone()

ALCARECOTkAlMuonIsolatedHIGoodMuons = ALCARECOTkAlMuonIsolatedGoodMuons.clone()
ALCARECOTkAlMuonIsolatedHIRelCombIsoMuons = ALCARECOTkAlMuonIsolatedRelCombIsoMuons.clone(
    src = 'ALCARECOTkAlMuonIsolatedHIGoodMuons'
)

ALCARECOTkAlMuonIsolatedHI = ALCARECOTkAlMuonIsolated.clone(
    src = 'hiGeneralTracks'
)

ALCARECOTkAlMuonIsolatedHI.GlobalSelector.muonSource = 'ALCARECOTkAlMuonIsolatedHIRelCombIsoMuons'
# Isolation is shifted to the muon preselection, and then applied intrinsically if applyGlobalMuonFilter = True
ALCARECOTkAlMuonIsolatedHI.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlMuonIsolatedHI.GlobalSelector.minJetDeltaR = 0.1
ALCARECOTkAlMuonIsolatedHI.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlMuonIsolatedHI.GlobalSelector.jetIsoSource = cms.InputTag("iterativeConePu5CaloJets")
ALCARECOTkAlMuonIsolatedHI.GlobalSelector.jetCountSource = cms.InputTag("iterativeConePu5CaloJets")

ALCARECOTkAlMuonIsolatedHI.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlMuonIsolatedHI.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlMuonIsolatedHI.TwoBodyDecaySelector.applyAcoplanarityFilter = False

seqALCARECOTkAlMuonIsolatedHI = cms.Sequence(ALCARECOTkAlMuonIsolatedHIHLT+ALCARECOTkAlMuonIsolatedHIDCSFilter+ALCARECOTkAlMuonIsolatedHIGoodMuons+ALCARECOTkAlMuonIsolatedHIRelCombIsoMuons+ALCARECOTkAlMuonIsolatedHI)
