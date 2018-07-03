# AlCaReco for track based alignment using isolated muon tracks - relaxed cuts for PbPb collisions
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlMuonIsolated_cff import *

ALCARECOTkAlMuonIsolatedHIHLT = ALCARECOTkAlMuonIsolatedHLT.clone(
    eventSetupPathsKey = 'TkAlMuonIsolatedHI'
    )

ALCARECOTkAlMuonIsolatedHIDCSFilter = ALCARECOTkAlMuonIsolatedDCSFilter.clone()

ALCARECOTkAlMuonIsolatedHI = ALCARECOTkAlMuonIsolated.clone(
    src = 'hiGeneralTracks'
)

# Isolation is shifted to the muon preselection, and then applied intrinsically if applyGlobalMuonFilter = True
ALCARECOTkAlMuonIsolatedHI.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlMuonIsolatedHI.GlobalSelector.minJetDeltaR = 0.0 #pp version has 0.1
ALCARECOTkAlMuonIsolatedHI.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlMuonIsolatedHI.GlobalSelector.jetIsoSource = cms.InputTag("iterativeConePu5CaloJets")
ALCARECOTkAlMuonIsolatedHI.GlobalSelector.jetCountSource = cms.InputTag("iterativeConePu5CaloJets")

ALCARECOTkAlMuonIsolatedHI.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlMuonIsolatedHI.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlMuonIsolatedHI.TwoBodyDecaySelector.applyAcoplanarityFilter = False

seqALCARECOTkAlMuonIsolatedHI = cms.Sequence(ALCARECOTkAlMuonIsolatedHIHLT
                                             +ALCARECOTkAlMuonIsolatedHIDCSFilter
                                             +ALCARECOTkAlMuonIsolatedHI
                                             )
