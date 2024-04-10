import FWCore.ParameterSet.Config as cms

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi


##
## ALCARECOTkAlMuonIsolated selection
##

## First select goodId + isolated muons
ALCARECOTkAlMuonIsolatedGoodMuons = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.TkAlGoodIdMuonSelector.clone()
ALCARECOTkAlMuonIsolatedRelCombIsoMuons = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.TkAlRelCombIsoMuonSelector.clone(
    src = 'ALCARECOTkAlMuonIsolatedGoodMuons'
)

## Then select their tracks with additional cuts
ALCARECOTkAlMuonIsolated = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    filter = True, ##do not store empty events
    applyBasicCuts = True,
    ptMin = 2.0, ##GeV 
    etaMin = -3.5,
    etaMax = 3.5,
    nHitMin = 0
)
ALCARECOTkAlMuonIsolated.GlobalSelector.muonSource = 'ALCARECOTkAlMuonIsolatedRelCombIsoMuons'
# Isolation is shifted to the muon preselection, and then applied intrinsically if applyGlobalMuonFilter = True
ALCARECOTkAlMuonIsolated.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlMuonIsolated.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlMuonIsolated.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlMuonIsolated.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlMuonIsolated.TwoBodyDecaySelector.applyAcoplanarityFilter = False

## Define ALCARECO sequence
mySeqALCARECOTkAlMuonIsolated = cms.Sequence(ALCARECOTkAlMuonIsolatedGoodMuons*ALCARECOTkAlMuonIsolatedRelCombIsoMuons*ALCARECOTkAlMuonIsolated)



##
## Good Primary Vertex Selection
##
goodPVs = cms.EDFilter("VertexSelector",
    src = cms.InputTag('offlinePrimaryVertices'),
    cut = cms.string('ndof>4 &'
                     'abs(z)<24 &'
                     '!isFake &'
                     'position.Rho<2'
    ),
)
oneGoodPVSelection = cms.EDFilter("VertexCountFilter",
    src = cms.InputTag('goodPVs'),
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(99999),
    
)
seqVertexSelection = cms.Sequence(goodPVs*oneGoodPVSelection)







