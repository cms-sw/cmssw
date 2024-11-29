import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from DPGAnalysis.MuonTools.common_cff import *


from PhysicsTools.NanoAOD.l1trig_cff import *

from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import (
    simpleCandidateFlatTableProducer,
)

from Validation.RecoMuon.muonValidationHLT_cff import *

from DPGAnalysis.MuonTools.dtSegmentFlatTableProducer_cfi import (
    dtSegmentFlatTableProducer,
)

hltDtSegmentFlatTable = dtSegmentFlatTableProducer.clone(
    name="dtSegment",
    src="hltDt4DSegments",
    doc="DT segment information",
    variables=cms.PSet(
        seg4D_hasPhi=Var("hasPhi()", bool, doc="has segment phi view - bool"),
        seg4D_hasZed=Var("hasZed()", bool, doc="has segment zed view - bool"),
        seg4D_posLoc_x=Var(
            "localPosition().x()", float, doc="position x in local coordinates - cm"
        ),
        seg4D_posLoc_y=Var(
            "localPosition().y()", float, doc="position y in local coordinates - cm"
        ),
        seg4D_posLoc_z=Var(
            "localPosition().z()", float, doc="position z in local coordinates - cm"
        ),
        seg4D_dirLoc_x=Var(
            "localDirection().x()", float, doc="direction x in local coordinates"
        ),
        seg4D_dirLoc_y=Var(
            "localDirection().y()", float, doc="direction y in local coordinates"
        ),
        seg4D_dirLoc_z=Var(
            "localDirection().z()", float, doc="direction z in local coordinates"
        ),
        seg2D_phi_t0=Var(
            f"? hasPhi() ? phiSegment().t0() : {defaults.FLOAT}",
            float,
            doc="t0 from segments with phi view - ns",
        ),
        seg2D_phi_nHits=Var(
            f"? hasPhi() ? phiSegment().specificRecHits().size() : 0",
            "int16",
            doc="# hits in phi view - [0:8] range",
        ),
        seg2D_phi_vDrift=Var(
            f"? hasPhi() ? phiSegment().vDrift() : {defaults.FLOAT_POS}",
            float,
            doc="v_drift from segments with phi view",
        ),
        seg2D_phi_normChi2=Var(
            f"? hasPhi() ? (phiSegment().chi2() / phiSegment().degreesOfFreedom()) : {defaults.FLOAT_POS}",
            float,
            doc="chi2/n.d.o.f. from segments with phi view",
        ),
        seg2D_z_t0=Var(
            f"? hasZed() ? zSegment().t0() : {defaults.FLOAT}",
            float,
            doc="t0 from segments with z view - ns",
        ),
        seg2D_z_nHits=Var(
            f"? hasZed() ? zSegment().specificRecHits().size() : 0",
            "int16",
            doc="# hits in z view - [0:4] range",
        ),
        seg2D_z_normChi2=Var(
            f"? hasZed() ? (zSegment().chi2() / zSegment().degreesOfFreedom()) : {defaults.FLOAT_POS}",
            float,
            doc="chi2/n.d.o.f. from segments with z view",
        ),
    ),
    detIdVariables=cms.PSet(
        wheel=DetIdVar("wheel()", "int16", doc="wheel  -  [-2:2] range"),
        sector=DetIdVar(
            "sector()",
            "int16",
            doc="sector - [1:14] range"
            "<br />sector 13 used for the second MB4 of sector 4"
            "<br />sector 14 used for the second MB4 of sector 10",
        ),
        station=DetIdVar("station()", "int16", doc="station - [1:4] range"),
    ),
    globalPosVariables=cms.PSet(
        seg4D_posGlb_phi=GlobGeomVar(
            "phi().value()", doc="position phi in global coordinates - radians [-pi:pi]"
        ),
        seg4D_posGlb_eta=GlobGeomVar("eta()", doc="position eta in global coordinates"),
    ),
    globalDirVariables=cms.PSet(
        seg4D_dirGlb_phi=GlobGeomVar(
            "phi().value()",
            doc="direction phi in global coordinates - radians [-pi:pi]",
        ),
        seg4D_dirGlb_eta=GlobGeomVar(
            "eta()", doc="direction eta in global coordinates"
        ),
    ),
)

from DPGAnalysis.MuonTools.rpcRecHitFlatTableProducer_cfi import (
    rpcRecHitFlatTableProducer,
)

hltRpcRecHitFlatTable = rpcRecHitFlatTableProducer.clone(
    name="rpcRecHit",
    src="hltRpcRecHits",
    doc="RPC rec-hit information",
    variables=cms.PSet(
        bx=Var("BunchX()", int, doc="bunch crossing number"),
        time=Var("time()", float, doc="time information in ns"),
        firstClusterStrip=Var(
            "firstClusterStrip()", "int16", doc="lowest-numbered strip in the cluster"
        ),
        clusterSize=Var(
            "clusterSize()", "int16", doc="number of strips in the cluster"
        ),
        coordX=Var(
            "localPosition().x()", float, doc="position x in local coordinates - cm"
        ),
        coordY=Var(
            "localPosition().y()", float, doc="position y in local coordinates - cm"
        ),
        coordZ=Var(
            "localPosition().z()", float, doc="position z in local coordinates - cm"
        ),
    ),
    detIdVariables=cms.PSet(
        region=DetIdVar("region()", "int16", doc="0: barrel, +-1: endcap"),
        ring=DetIdVar(
            "ring()",
            "int16",
            doc="ring id:"
            "<br />wheel number in barrel (from -2 to +2)"
            "<br />ring number in endcap (from 1 to 3)",
        ),
        station=DetIdVar(
            "station()",
            "int16",
            doc="chambers at same R in barrel, chambers at same Z ion endcap",
        ),
        layer=DetIdVar(
            "layer()",
            "int16",
            doc="layer id:"
            "<br />in station 1 and 2 for barrel, we have two layers of chambers:"
            "<br />layer 1 is the inner chamber and layer 2 is the outer chamber",
        ),
        sector=DetIdVar("sector()", "int16", doc="group of chambers at same phi"),
        subsector=DetIdVar(
            "subsector()",
            "int16",
            doc="Some sectors are divided along the phi direction in subsectors "
            "(from 1 to 4 in Barrel, from 1 to 6 in Endcap)",
        ),
        roll=DetIdVar(
            "roll()",
            "int16",
            doc="roll id (also known as eta partition):"
            "<br />each chamber is divided along the strip direction",
        ),
        rawId=DetIdVar("rawId()", "uint", doc="unique detector unit ID"),
    ),
)


from DPGAnalysis.MuonTools.gemRecHitFlatTableProducer_cfi import (
    gemRecHitFlatTableProducer,
)

hltGemRecHitFlatTable = gemRecHitFlatTableProducer.clone(
    name="gemRecHit",
    src="hltFGmRecHits",
    doc="GEM rec-hit information",
    variables=cms.PSet(
        bx=Var("BunchX()", int, doc="bunch crossing number"),
        clusterSize=Var(
            "clusterSize()", "int16", doc="number of strips in the cluster"
        ),
        loc_x=Var(
            "localPosition().x()", float, doc="hit position x in local coordinates - cm"
        ),
        firstClusterStrip=Var(
            "firstClusterStrip()", "int16", doc="lowest-numbered strip in the cluster"
        ),
        loc_phi=Var(
            "localPosition().phi().value()",
            float,
            doc="hit position phi in local coordinates - rad",
        ),
        loc_y=Var(
            "localPosition().y()", float, doc="hit position y in local coordinates - cm"
        ),
        loc_z=Var(
            "localPosition().z()", float, doc="hit position z in local coordinates - cm"
        ),
    ),
    detIdVariables=cms.PSet(
        roll=DetIdVar(
            "roll()",
            "int16",
            doc="roll id, also known as eta partition:"
            "<br />(partitions numbered from 1 to 8)",
        ),
        region=DetIdVar(
            "region()",
            "int16",
            doc="GE11 region where the hit is reconstructed"
            "<br />(int, positive endcap: +1, negative endcap: -1)",
        ),
        chamber=DetIdVar(
            "chamber()",
            "int16",
            doc="GE11 superchamber where the hit is reconstructed"
            "<br />(chambers numbered from 0 to 35)",
        ),
        layer=DetIdVar(
            "layer()",
            "int16",
            doc="GE11 layer where the hit is reconstructed"
            "<br />(layer1: 1, layer2: 2)",
        ),
    ),
    globalPosVariables=cms.PSet(
        g_r=GlobGeomVar("perp()", doc="hit position r in global coordinates - cm"),
        g_phi=GlobGeomVar(
            "phi().value()",
            doc="hit position phi in global coordinates -  radians [-pi:pi]",
        ),
        g_x=GlobGeomVar("x()", doc="hit position x in global coordinates - cm"),
        g_y=GlobGeomVar("y()", doc="hit position y in global coordinates - cm"),
        g_z=GlobGeomVar("z()", doc="hit position z in global coordinates - cm"),
    ),
)

from DPGAnalysis.MuonTools.gemSegmentFlatTableProducer_cfi import (
    gemSegmentFlatTableProducer,
)

hltGemSegmentFlatTable = gemSegmentFlatTableProducer.clone(
    name="gemSegment",
    src="hltGemSegments",
    doc="GEM segment information",
    variables=cms.PSet(
        chi2=Var("chi2()", int, doc="chi2 from segment fit"),
        bx=Var("bunchX()", int, doc="bunch crossing number"),
        posLoc_x=Var(
            "localPosition().x()", float, doc="position x in local coordinates - cm"
        ),
        posLoc_y=Var(
            "localPosition().y()", float, doc="position y in local coordinates - cm"
        ),
        posLoc_z=Var(
            "localPosition().z()", float, doc="position z in local coordinates - cm"
        ),
        dirLoc_x=Var(
            "localDirection().x()", float, doc="direction x in local coordinates"
        ),
        dirLoc_y=Var(
            "localDirection().y()", float, doc="direction y in local coordinates"
        ),
        dirLoc_z=Var(
            "localDirection().z()", float, doc="direction z in local coordinates"
        ),
    ),
    detIdVariables=cms.PSet(
        region=DetIdVar(
            "region()",
            "int16",
            doc="GE11 region where the hit is reconstructed"
            "<br />(int, positive endcap: +1, negative endcap: -1)",
        ),
        ring=DetIdVar("ring()", "int16", doc=""),
        station=DetIdVar(
            "station()", "int16", doc="GEM station <br />(always 1 for GE1/1)"
        ),
        chamber=DetIdVar(
            "chamber()",
            "int16",
            doc="GE11 superchamber where the hit is reconstructed"
            "<br />(chambers numbered from 0 to 35)",
        ),
    ),
    globalPosVariables=cms.PSet(
        posGlb_x=GlobGeomVar("x()", doc="position x in global coordinates - cm"),
        posGlb_y=GlobGeomVar("y()", doc="position y in global coordinates - cm"),
        posGlb_z=GlobGeomVar("z()", doc="position z in global coordinates - cm"),
        posGlb_phi=GlobGeomVar(
            "phi().value()", doc="position phi in global coordinates - radians [-pi:pi]"
        ),
        posGlb_eta=GlobGeomVar("eta()", doc="position eta in global coordinates"),
    ),
    globalDirVariables=cms.PSet(
        dirGlb_phi=GlobGeomVar(
            "phi().value()",
            doc="direction phi in global coordinates - radians [-pi:pi]",
        ),
        dirGlb_eta=GlobGeomVar("eta()", doc="direction eta in global coordinates"),
    ),
)

hltLocalRecoMuon_seq = cms.Sequence(
    hltDtSegmentFlatTable
    + hltRpcRecHitFlatTable
    + hltGemRecHitFlatTable
    + hltGemSegmentFlatTable
)

# L1Tk Muons
l1TkMuTable = cms.EDProducer(
    "SimpleTriggerL1TkMuonFlatTableProducer",
    src=cms.InputTag("l1tTkMuonsGmt"),
    cut=cms.string(""),
    name=cms.string("L1TkMu"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("phPt()", "float", doc="Physics pt"),
        eta=Var("phEta()", "float", doc="#eta"),
        phi=Var("phPhi()", "float", doc="#phi (rad)"),
        dXY=Var("phD0()", "float", doc="dXY (cm)"),
        dZ=Var("phZ0()", "float", doc="dZ (cm)"),
    ),
)

# L2 offline seeds
l2SeedTable = cms.EDProducer(
    "SimpleTrajectorySeedFlatTableProducer",
    src=cms.InputTag("hltL2OfflineMuonSeeds"),
    cut=cms.string(""),
    name=cms.string("l2_seed_offline"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("startingState().pt()", "float", doc="p_T (GeV)"),
        nHits=Var("nHits()", "int16", doc=""),
        localX=Var(
            "startingState().parameters().position().x()",
            "float",
            doc="local x of the seed",
        ),
        localY=Var(
            "startingState().parameters().position().y()",
            "float",
            doc="local y of the seed",
        ),
    ),
)

# L2 seeds from L1Tk Muons
l2SeedFromL1TkMuonTable = cms.EDProducer(
    "SimpleL2MuonTrajectorySeedFlatTableProducer",
    src=cms.InputTag("hltL2MuonSeedsFromL1TkMuon"),
    cut=cms.string(""),
    name=cms.string("phase2_l2_seed"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("startingState().pt()", "float", doc="p_T (GeV)"),
        nHits=Var("nHits()", "int16", doc=""),
        localX=Var(
            "startingState().parameters().position().x()",
            "float",
            doc="local x of the seed",
        ),
        localY=Var(
            "startingState().parameters().position().y()",
            "float",
            doc="local y of the seed",
        ),
    ),
)

# L2 standalone muons
l2MuTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src=cms.InputTag("hltL2MuonsFromL1TkMuon"),
    cut=cms.string(""),
    name=cms.string("l2_mu"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("pt()", "float", doc="p_T (GeV)"),
        eta=Var("eta()", "float", doc="#eta"),
        phi=Var("phi()", "float", doc="#phi (rad)"),
        dXY=Var("dxy()", "float", doc="dXY (cm)"),
        dZ=Var("dz()", "float", doc="dZ (cm)"),
        t0=Var("t0()", "float", doc="t0 (ns)"),
        nPixelHits=Var("hitPattern().numberOfValidPixelHits()", "int16", doc=""),
        nTrkLays=Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc=""),
        nMuHits=Var("hitPattern().numberOfValidMuonHits()", "int16", doc=""),
    ),
)

# L2 standalone muons updated at vertex
l2MuTableVtx = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src=cms.InputTag("hltL2MuonsFromL1TkMuon", "UpdatedAtVtx"),
    cut=cms.string(""),
    name=cms.string("l2_mu_vtx"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("pt()", "float", doc="p_T (GeV)"),
        eta=Var("eta()", "float", doc="#eta"),
        phi=Var("phi()", "float", doc="#phi (rad)"),
        dXY=Var("dxy()", "float", doc="dXY (cm)"),
        dZ=Var("dz()", "float", doc="dZ (cm)"),
        t0=Var("t0()", "float", doc="t0 (ns)"),
        nPixelHits=Var("hitPattern().numberOfValidPixelHits()", "int16", doc=""),
        nTrkLays=Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc=""),
        nMuHits=Var("hitPattern().numberOfValidMuonHits()", "int16", doc=""),
    ),
)

# L3 IO inner tracks
l3TkIOTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src=cms.InputTag("hltIter2Phase2L3FromL1TkMuonMerged"),
    cut=cms.string(""),
    name=cms.string("l3_tk_IO"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("pt()", "float", doc="p_T (GeV)"),
        eta=Var("eta()", "float", doc="#eta"),
        phi=Var("phi()", "float", doc="#phi (rad)"),
        dXY=Var("dxy()", "float", doc="dXY (cm)"),
        dZ=Var("dz()", "float", doc="dZ (cm)"),
        t0=Var("t0()", "float", doc="t0 (ns)"),
        nPixelHits=Var("hitPattern().numberOfValidPixelHits()", "int16", doc=""),
        nTrkLays=Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc=""),
        nMuHits=Var("hitPattern().numberOfValidMuonHits()", "int16", doc=""),
    ),
)

# L3 OI inner tracks
l3TkOITable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src=cms.InputTag("hltPhase2L3OIMuonTrackSelectionHighPurity"),
    cut=cms.string(""),
    name=cms.string("l3_tk_OI"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("pt()", "float", doc="p_T (GeV)"),
        eta=Var("eta()", "float", doc="#eta"),
        phi=Var("phi()", "float", doc="#phi (rad)"),
        dXY=Var("dxy()", "float", doc="dXY (cm)"),
        dZ=Var("dz()", "float", doc="dZ (cm)"),
        t0=Var("t0()", "float", doc="t0 (ns)"),
        nPixelHits=Var("hitPattern().numberOfValidPixelHits()", "int16", doc=""),
        nTrkLays=Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc=""),
        nMuHits=Var("hitPattern().numberOfValidMuonHits()", "int16", doc=""),
    ),
)

# L3 tracks merged
l3TkMergedTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src=cms.InputTag("hltPhase2L3MuonMerged"),
    cut=cms.string(""),
    name=cms.string("l3_tk_merged"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("pt()", "float", doc="p_T (GeV)"),
        eta=Var("eta()", "float", doc="#eta"),
        phi=Var("phi()", "float", doc="#phi (rad)"),
        dXY=Var("dxy()", "float", doc="dXY (cm)"),
        dZ=Var("dz()", "float", doc="dZ (cm)"),
        t0=Var("t0()", "float", doc="t0 (ns)"),
        nPixelHits=Var("hitPattern().numberOfValidPixelHits()", "int16", doc=""),
        nTrkLays=Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc=""),
        nMuHits=Var("hitPattern().numberOfValidMuonHits()", "int16", doc=""),
    ),
)

# L3 global muons
l3GlbMuTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src=cms.InputTag("hltPhase2L3GlbMuon"),
    cut=cms.string(""),
    name=cms.string("l3_mu_global"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("pt()", "float", doc="p_T (GeV)"),
        eta=Var("eta()", "float", doc="#eta"),
        phi=Var("phi()", "float", doc="#phi (rad)"),
        dXY=Var("dxy()", "float", doc="dXY (cm)"),
        dZ=Var("dz()", "float", doc="dZ (cm)"),
        t0=Var("t0()", "float", doc="t0 (ns)"),
        nPixelHits=Var("hitPattern().numberOfValidPixelHits()", "int16", doc=""),
        nTrkLays=Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc=""),
        nMuHits=Var("hitPattern().numberOfValidMuonHits()", "int16", doc=""),
    ),
)

# L3 Muons no ID (tracks)
l3MuTkNoIdTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src=cms.InputTag("hltPhase2L3MuonNoIdTracks"),
    cut=cms.string(""),
    name=cms.string("l3_mu_no_ID"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("pt()", "float", doc="p_T (GeV)"),
        eta=Var("eta()", "float", doc="#eta"),
        phi=Var("phi()", "float", doc="#phi (rad)"),
        dXY=Var("dxy()", "float", doc="dXY (cm)"),
        dZ=Var("dz()", "float", doc="dZ (cm)"),
        t0=Var("t0()", "float", doc="t0 (ns)"),
        nPixelHits=Var("hitPattern().numberOfValidPixelHits()", "int16", doc=""),
        nTrkLays=Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc=""),
        nMuHits=Var("hitPattern().numberOfValidMuonHits()", "int16", doc=""),
    ),
)

# L3 Muons ID (tracks)
l3MuTkIdTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src=cms.InputTag("hltPhase2L3MuonIdTracks"),
    cut=cms.string(""),
    name=cms.string("l3_mu_ID"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("pt()", "float", doc="p_T (GeV)"),
        eta=Var("eta()", "float", doc="#eta"),
        phi=Var("phi()", "float", doc="#phi (rad)"),
        dXY=Var("dxy()", "float", doc="dXY (cm)"),
        dZ=Var("dz()", "float", doc="dZ (cm)"),
        t0=Var("t0()", "float", doc="t0 (ns)"),
        nPixelHits=Var("hitPattern().numberOfValidPixelHits()", "int16", doc=""),
        nTrkLays=Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc=""),
        nMuHits=Var("hitPattern().numberOfValidMuonHits()", "int16", doc=""),
    ),
)

# L2 muons to reuse (IO first)
l2MuToReuseTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src=cms.InputTag("hltPhase2L3MuonFilter", "L2MuToReuse"),
    cut=cms.string(""),
    name=cms.string("l2_mu_to_reuse"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("pt()", "float", doc="p_T (GeV)"),
        eta=Var("eta()", "float", doc="#eta"),
        phi=Var("phi()", "float", doc="#phi (rad)"),
        dXY=Var("dxy()", "float", doc="dXY (cm)"),
        dZ=Var("dz()", "float", doc="dZ (cm)"),
        t0=Var("t0()", "float", doc="t0 (ns)"),
        nMuHits=Var("hitPattern().numberOfValidMuonHits()", "int16", doc=""),
    ),
)

# L3 IO tracks filtered (IO first)
l3TkIOFilteredTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src=cms.InputTag("hltPhase2L3MuonFilter:L3IOTracksFiltered"),
    cut=cms.string(""),
    name=cms.string("l3_tk_IO_filtered"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("pt()", "float", doc="p_T (GeV)"),
        eta=Var("eta()", "float", doc="#eta"),
        phi=Var("phi()", "float", doc="#phi (rad)"),
        dXY=Var("dxy()", "float", doc="dXY (cm)"),
        dZ=Var("dz()", "float", doc="dZ (cm)"),
        t0=Var("t0()", "float", doc="t0 (ns)"),
        nPixelHits=Var("hitPattern().numberOfValidPixelHits()", "int16", doc=""),
        nTrkLays=Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc=""),
        nMuHits=Var("hitPattern().numberOfValidMuonHits()", "int16", doc=""),
    ),
)

# L1 Tracker Muons to reuse (OI first)
l1TkMuToReuseTable = cms.EDProducer(
    "SimpleTriggerL1TkMuonFlatTableProducer",
    src=cms.InputTag("hltPhase2L3MuonFilter", "L1TkMuToReuse"),
    cut=cms.string(""),
    name=cms.string("L1TkMu_to_reuse"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("phPt()", "float", doc="Physics pt"),
        eta=Var("phEta()", "float", doc="#eta"),
        phi=Var("phPhi()", "float", doc="#phi (rad)"),
        dXY=Var("phD0()", "float", doc="dXY (cm)"),
        dZ=Var("phZ0()", "float", doc="dZ (cm)"),
    ),
)

# L3 OI tracks filtered (OI first)
l3TkOIFilteredTable = cms.EDProducer(
    "SimpleTriggerTrackFlatTableProducer",
    src=cms.InputTag("hltPhase2L3MuonFilter:L3OITracksFiltered"),
    cut=cms.string(""),
    name=cms.string("l3_tk_OI_filtered"),
    doc=cms.string(""),
    extension=cms.bool(False),
    variables=cms.PSet(
        pt=Var("pt()", "float", doc="p_T (GeV)"),
        eta=Var("eta()", "float", doc="#eta"),
        phi=Var("phi()", "float", doc="#phi (rad)"),
        dXY=Var("dxy()", "float", doc="dXY (cm)"),
        dZ=Var("dz()", "float", doc="dZ (cm)"),
        t0=Var("t0()", "float", doc="t0 (ns)"),
        nPixelHits=Var("hitPattern().numberOfValidPixelHits()", "int16", doc=""),
        nTrkLays=Var("hitPattern().trackerLayersWithMeasurement()", "int16", doc=""),
        nMuHits=Var("hitPattern().numberOfValidMuonHits()", "int16", doc=""),
    ),
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
    variables=cms.PSet(
        pt=Var("startingState().pt()", "float", doc="p_T (GeV)"),
        nHits=Var(
            "nHits()", "int16", doc="number of DT/CSC segments propagated to the seed"
        ),
        eta=Var("l1TkMu().phEta()", "float", doc="associated L1TkMu #eta"),
        phi=Var("l1TkMu().phPhi()", "float", doc="associated L1TkMu #phi"),
        localX=Var(
            "startingState().parameters().position().x()",
            "float",
            doc="local x of the seed",
        ),
        localY=Var(
            "startingState().parameters().position().y()",
            "float",
            doc="local y of the seed",
        ),
    ),
)
phase2L2AndL3Muons.toReplaceWith(
    hltMuonTriggerProducers, hltMuonTriggerProducersIOFirst
)

from Configuration.ProcessModifiers.phase2L3MuonsOIFirst_cff import phase2L3MuonsOIFirst

(phase2L2AndL3Muons & phase2L3MuonsOIFirst).toModify(
    l2SeedFromL1TkMuonTable,
    variables=cms.PSet(
        pt=Var("startingState().pt()", "float", doc="p_T (GeV)"),
        nHits=Var(
            "nHits()", "int16", doc="number of DT/CSC segments propagated to the seed"
        ),
        eta=Var("l1TkMu().phEta()", "float", doc="associated L1TkMu #eta"),
        phi=Var("l1TkMu().phPhi()", "float", doc="associated L1TkMu #phi"),
        localX=Var(
            "startingState().parameters().position().x()",
            "float",
            doc="local x of the seed",
        ),
        localY=Var(
            "startingState().parameters().position().y()",
            "float",
            doc="local y of the seed",
        ),
    ),
)
(phase2L2AndL3Muons & phase2L3MuonsOIFirst).toReplaceWith(
    hltMuonTriggerProducers, hltMuonTriggerProducersOIFirst
)
