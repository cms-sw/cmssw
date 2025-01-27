import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from DPGAnalysis.MuonTools.common_cff import *


from PhysicsTools.NanoAOD.l1trig_cff import *

from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer

from Validation.RecoMuon.muonValidationHLT_cff import *

from DPGAnalysis.MuonTools.nano_mu_local_reco_cff import dtSegmentFlatTable as dtSegmentFlatTable_

hltDtSegmentFlatTable = dtSegmentFlatTable_.clone(
    name = "dtSegment",
    src = "hltDt4DSegments",
    doc = "DT segment information"
)

from DPGAnalysis.MuonTools.cscSegmentFlatTableProducer_cfi import cscSegmentFlatTableProducer

hltCscSegmentFlatTable = cscSegmentFlatTableProducer.clone(
    name = "cscSegment",
    src = "hltCscSegments",
    doc = "CSC segment information",
    variables = cms.PSet(
        degreesOfFreedom = Var("degreesOfFreedom()", int, doc = "Degrees of freedom of the 4D CSC segment"),
        nHits = Var("nRecHits()", int, doc = "Number of recHits used to build the segment"),
        posLoc_x = Var(
            "localPosition().x()", float, doc = "position x in local coordinates - cm"
        ),
        posLoc_y = Var(
            "localPosition().y()", float, doc = "position y in local coordinates - cm"
        ),
        posLoc_z = Var(
            "localPosition().z()", float, doc = "position z in local coordinates - cm"
        ),
        dirLoc_x = Var(
            "localDirection().x()", float, doc = "direction x in local coordinates"
        ),
        dirLoc_y = Var(
            "localDirection().y()", float, doc = "direction y in local coordinates"
        ),
        dirLoc_z = Var(
            "localDirection().z()", float, doc = "direction z in local coordinates"
        ),
        normChi2 = Var(
            "chi2() / degreesOfFreedom()", float, doc = "chi2/n.d.o.f. for all segments"
        )
    ),    
    detIdVariables = cms.PSet(
        endcap = DetIdVar("endcap()", int, doc = "Endcap - 1 Forward (+Z), 2 Backward (-Z)"),
        layer = DetIdVar("layer()", int, doc = "Layer [1:6]"),
        chamber = DetIdVar("chamber()", int, doc = "Chamber [1:36]"),
        ring = DetIdVar("ring()", int, doc = "Ring [1:4]"),
        station = DetIdVar("station()", int, doc = "Station [1-4]")
    ),
    globalPosVariables = cms.PSet(
        posGlb_phi = GlobGeomVar(
            "phi().value()", doc = "position phi in global coordinates - radians [-pi:pi]"
        ),
        posGlb_eta = GlobGeomVar("eta()", doc = "position eta in global coordinates")
    ),
    globalDirVariables = cms.PSet(
        dirGlb_phi = GlobGeomVar(
            "phi().value()",
            doc = "direction phi in global coordinates - radians [-pi:pi]",
        ),
        dirGlb_eta = GlobGeomVar(
            "eta()", doc = "direction eta in global coordinates"
        )
    )
)

from DPGAnalysis.MuonTools.nano_mu_local_reco_cff import rpcRecHitFlatTable as rpcRecHitFlatTable_

hltRpcRecHitFlatTable = rpcRecHitFlatTable_.clone(
    name = "rpcRecHit",
    src = "hltRpcRecHits",
    doc = "RPC rec-hit information"
)

from DPGAnalysis.MuonTools.nano_mu_local_reco_cff import gemRecHitFlatTable as gemRecHitFlatTable_

hltGemRecHitFlatTable = gemRecHitFlatTable_.clone(
    name = "gemRecHit",
    src = "hltGemRecHits",
    doc = "GEM rec-hit information"
)

from DPGAnalysis.MuonTools.nano_mu_local_reco_cff import gemSegmentFlatTable as gemSegmentFlatTable_

hltGemSegmentFlatTable = gemSegmentFlatTable_.clone(
    name = "gemSegment",
    src = "hltGemSegments",
    doc = "GEM segment information"
)

hltLocalRecoMuon_seq = cms.Sequence(
    hltDtSegmentFlatTable
    + hltCscSegmentFlatTable
    + hltRpcRecHitFlatTable
    + hltGemRecHitFlatTable
    + hltGemSegmentFlatTable
)

# L1Tk Muons
l1TkMuTable = cms.EDProducer(
    "SimpleTriggerL1TkMuonFlatTableProducer",
    src = cms.InputTag("l1tTkMuonsGmt"),
    cut = cms.string(""),
    name = cms.string("L1TkMu"),
    doc = cms.string(""),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("phPt()", "float", doc = "Physics pt"),
        eta = Var("phEta()", "float", doc = "#eta"),
        phi = Var("phPhi()", "float", doc = "#phi (rad)"),
        dXY = Var("phD0()", "float", doc = "dXY (cm)"),
        dZ = Var("phZ0()", "float", doc = "dZ (cm)")
    )
)

# L2 offline seeds
l2SeedTable = cms.EDProducer(
    "SimpleTrajectorySeedFlatTableProducer",
    src = cms.InputTag("hltL2OfflineMuonSeeds"),
    cut = cms.string(""),
    name = cms.string("l2_seed_offline"),
    doc = cms.string(""),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("startingState().pt()", "float", doc = "p_T (GeV)"),
        nHits = Var("nHits()", "int16", doc = ""),
        localX = Var(
            "startingState().parameters().position().x()",
            "float",
            doc = "local x of the seed",
        ),
        localY = Var(
            "startingState().parameters().position().y()",
            "float",
            doc = "local y of the seed",
        )
    )
)

# L2 seeds from L1Tk Muons
l2SeedFromL1TkMuonTable = cms.EDProducer(
    "SimpleL2MuonTrajectorySeedFlatTableProducer",
    src = cms.InputTag("hltL2MuonSeedsFromL1TkMuon"),
    cut = cms.string(""),
    name = cms.string("phase2_l2_seed"),
    doc = cms.string(""),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("startingState().pt()", "float", doc = "p_T (GeV)"),
        nHits = Var("nHits()", "int16", doc = ""),
        localX = Var(
            "startingState().parameters().position().x()",
            "float",
            doc = "local x of the seed",
        ),
        localY = Var(
            "startingState().parameters().position().y()",
            "float",
            doc = "local y of the seed",
        )
    )
)

# L2 standalone muons
l2MuTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src = cms.InputTag("hltL2MuonsFromL1TkMuon"),
    cut = cms.string(""),
    name = cms.string("l2_mu"),
    doc = cms.string("Standalone Muon tracks"),
    extension = cms.bool(False),
    variables = cms.PSet(
        pt = Var("pt()", "float", doc = "p_T (GeV)"),
        eta = Var("eta()", "float", doc = "#eta"),
        phi = Var("phi()", "float", doc = "#phi (rad)"),
        dXY = Var("dxy()", "float", doc = "dXY (cm)"),
        dZ = Var("dz()", "float", doc = "dZ (cm)"),
        t0 = Var("t0()", "float", doc = "t0 (ns)"),
        nPixelHits = Var("hitPattern().numberOfValidPixelHits()", "int16", doc = ""),
        nTrkLays = Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc = ""),
        nMuHits = Var("hitPattern().numberOfValidMuonHits()", "int16", doc = "")
    )
)

# L2 standalone muons updated at vertex
l2MuTableVtx = l2MuTable.clone(
    src = cms.InputTag("hltL2MuonsFromL1TkMuon:UpdatedAtVtx"),
    name = cms.string("l2_mu_vtx"),
    doc = cms.string("Standalone Muon tracks updated at vertex")
)

# L3 IO inner tracks
l3TkIOTable = l2MuTable.clone(
    src = cms.InputTag("hltIter2Phase2L3FromL1TkMuonMerged"),
    name = cms.string("l3_tk_IO"),
    doc = cms.string("L3 Tracker Muon tracks Inside-Out")
)

# L3 OI inner tracks
l3TkOITable = l2MuTable.clone(
    src = cms.InputTag("hltPhase2L3OIMuonTrackSelectionHighPurity"),
    name = cms.string("l3_tk_OI"),
    doc = cms.string("L3 Tracker Muon tracks Outside-In")
)

# L3 tracks merged
l3TkMergedTable = l2MuTable.clone(
    src = cms.InputTag("hltPhase2L3MuonMerged"),
    name = cms.string("l3_tk_merged"),
    doc = cms.string("L3 Tracker Muon tracks merged (IO + OI)")
)

# L3 global muons
l3GlbMuTable = l2MuTable.clone(
    src = cms.InputTag("hltPhase2L3GlbMuon"),
    name = cms.string("l3_mu_global"),
    doc = cms.string("Global Muons (L3 Tracker Muon + Standalone)")
)

# L3 Muons no ID (tracks)
l3MuTkNoIdTable = l2MuTable.clone(
    src = cms.InputTag("hltPhase2L3MuonNoIdTracks"),
    name = cms.string("l3_mu_no_ID"),
    doc = cms.string("Muon tracks before ID")
)

# L3 Muons ID (tracks)
l3MuTkIdTable = l2MuTable.clone(
    src = cms.InputTag("hltPhase2L3MuonIdTracks"),
    name = cms.string("l3_mu_ID"),
    doc = cms.string("Muon tracks after ID")
)

# L2 muons to reuse (IO first)
l2MuToReuseTable = l2MuTable.clone(
    src = cms.InputTag("hltPhase2L3MuonFilter:L2MuToReuse"),
    name = cms.string("l2_mu_to_reuse"),
    doc = cms.string("Standlone Muon tracks to reuse (not matched with L3 Tracker Muon)")
)

# L3 IO tracks filtered (IO first)
l3TkIOFilteredTable = l2MuTable.clone(
    src = cms.InputTag("hltPhase2L3MuonFilter:L3IOTracksFiltered"),
    name = cms.string("l3_tk_IO_filtered"),
    doc = cms.string("L3 Tracker Muons Inside-Out filtered (quality cuts and match with Standalone)")
)

# L1 Tracker Muons to reuse (OI first)
l1TkMuToReuseTable = l1TkMuTable.clone(
    src = cms.InputTag("hltPhase2L3MuonFilter:L1TkMuToReuse"),
    name = cms.string("L1TkMu_to_reuse"),
    doc = cms.string("L1TkMuons not matched with L3 Tracker Muon OI")
)

# L3 OI tracks filtered (OI first)
l3TkOIFilteredTable = l2MuTable.clone(
    src = cms.InputTag("hltPhase2L3MuonFilter:L3OITracksFiltered"),
    name = cms.string("l3_tk_OI_filtered"),
    doc = cms.string("L3 Tracker Muons Outside-In filtered (quality cuts and match with L1TkMu)")
)

# The muon trigger producers sequence
hltMuonTriggerProducers = cms.Sequence(
    recoMuonValidationHLT_seq
    + hltLocalRecoMuon_seq
    + l1TkMuTable
    + l2SeedTable
    + l2SeedFromL1TkMuonTable
    + l2MuTable
    + l2MuTableVtx
    + l3TkIOTable
    + l3TkOITable
    + l3TkMergedTable
    + l3GlbMuTable
    + l3MuTkNoIdTable
    + l3MuTkIdTable
)

# The Phase-2 IO first muon trigger producers sequence
hltMuonTriggerProducersIOFirst = cms.Sequence(
    recoMuonValidationHLT_seq
    + hltLocalRecoMuon_seq
    + l1TkMuTable
    + l2SeedFromL1TkMuonTable
    + l2MuTable
    + l2MuTableVtx
    + l3TkIOTable
    + l2MuToReuseTable
    + l3TkIOFilteredTable
    + l3TkOITable
    + l3TkMergedTable
    + l3GlbMuTable
    + l3MuTkNoIdTable
    + l3MuTkIdTable
)

# The Phase-2 OI first muon trigger producers sequence
hltMuonTriggerProducersOIFirst = cms.Sequence(
    recoMuonValidationHLT_seq
    + hltLocalRecoMuon_seq
    + l1TkMuTable
    + l2SeedFromL1TkMuonTable
    + l2MuTable
    + l2MuTableVtx
    + l3TkOITable
    + l1TkMuToReuseTable
    + l3TkOIFilteredTable
    + l3TkIOTable
    + l3TkMergedTable
    + l3GlbMuTable
    + l3MuTkNoIdTable
    + l3MuTkIdTable
)

from Configuration.ProcessModifiers.phase2L2AndL3Muons_cff import phase2L2AndL3Muons

phase2L2AndL3Muons.toModify(
    l2SeedFromL1TkMuonTable,
    variables = cms.PSet(
        pt = Var("startingState().pt()", "float", doc = "p_T (GeV)"),
        nHits = Var(
            "nHits()", "int16", doc = "number of DT/CSC segments propagated to the seed"
        ),
        eta = Var("l1TkMu().phEta()", "float", doc = "associated L1TkMu #eta"),
        phi = Var("l1TkMu().phPhi()", "float", doc = "associated L1TkMu #phi"),
        localX = Var(
            "startingState().parameters().position().x()",
            "float",
            doc = "local x of the seed",
        ),
        localY = Var(
            "startingState().parameters().position().y()",
            "float",
            doc = "local y of the seed",
        )
    )
)
phase2L2AndL3Muons.toReplaceWith(
    hltMuonTriggerProducers, hltMuonTriggerProducersIOFirst
)

from Configuration.ProcessModifiers.phase2L3MuonsOIFirst_cff import phase2L3MuonsOIFirst

(phase2L2AndL3Muons & phase2L3MuonsOIFirst).toModify(
    l2SeedFromL1TkMuonTable,
    variables = cms.PSet(
        pt = Var("startingState().pt()", "float", doc = "p_T (GeV)"),
        nHits = Var(
            "nHits()", "int16", doc = "number of DT/CSC segments propagated to the seed"
        ),
        eta = Var("l1TkMu().phEta()", "float", doc = "associated L1TkMu #eta"),
        phi = Var("l1TkMu().phPhi()", "float", doc = "associated L1TkMu #phi"),
        localX = Var(
            "startingState().parameters().position().x()",
            "float",
            doc = "local x of the seed",
        ),
        localY = Var(
            "startingState().parameters().position().y()",
            "float",
            doc = "local y of the seed",
        )
    )
)
(phase2L2AndL3Muons & phase2L3MuonsOIFirst).toReplaceWith(
    hltMuonTriggerProducers, hltMuonTriggerProducersOIFirst
)
