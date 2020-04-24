import FWCore.ParameterSet.Config as cms



##
## ALCARECOTkAlMuonIsolated selection
##

## First select goodId + isolated muons
TkAlGoodIdMuonSelector = cms.EDFilter("MuonSelector",
    src = cms.InputTag('muons'),
    cut = cms.string('isGlobalMuon &'
                     'isTrackerMuon &'
                     'numberOfMatches > 1 &'
                     'globalTrack.hitPattern.numberOfValidMuonHits > 0 &'
                     'abs(eta) < 2.5 &'
                     'globalTrack.normalizedChi2 < 20.'),
    filter = cms.bool(True)
)
TkAlRelCombIsoMuonSelector = cms.EDFilter("MuonSelector",
    src = cms.InputTag(''),
    cut = cms.string('(isolationR03().sumPt + isolationR03().emEt + isolationR03().hadEt)/pt  < 0.15'),
    filter = cms.bool(True)
)
ALCARECOTkAlMuonIsolatedGoodMuons = TkAlGoodIdMuonSelector.clone()
ALCARECOTkAlMuonIsolatedRelCombIsoMuons = TkAlRelCombIsoMuonSelector.clone(src = 'ALCARECOTkAlMuonIsolatedGoodMuons')

## Then select their tracks with additional cuts
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
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







