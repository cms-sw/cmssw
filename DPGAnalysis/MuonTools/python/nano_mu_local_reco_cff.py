import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from DPGAnalysis.MuonTools.common_cff import *

from DPGAnalysis.MuonTools.dtSegmentFlatTableProducer_cfi import dtSegmentFlatTableProducer

dtSegmentFlatTable = dtSegmentFlatTableProducer.clone(
    name = "dtSegment",
    src =  "dt4DSegments",
    doc =  "DT segment information",

    variables = cms.PSet(
        seg4D_hasPhi = Var("hasPhi()", bool, doc = "has segment phi view - bool"),
        seg4D_hasZed = Var("hasZed()", bool, doc = "has segment zed view - bool"),
        seg4D_posLoc_x = Var("localPosition().x()", float, doc = "position x in local coordinates - cm"),
        seg4D_posLoc_y = Var("localPosition().y()", float, doc = "position y in local coordinates - cm"),
        seg4D_posLoc_z = Var("localPosition().z()", float, doc = "position z in local coordinates - cm"),
        seg4D_dirLoc_x = Var("localDirection().x()", float, doc = "direction x in local coordinates"),
        seg4D_dirLoc_y = Var("localDirection().y()", float, doc = "direction y in local coordinates"),
        seg4D_dirLoc_z = Var("localDirection().z()", float, doc = "direction z in local coordinates"),

        seg2D_phi_t0 = Var(f"? hasPhi() ? phiSegment().t0() : {defaults.FLOAT}", float, doc = "t0 from segments with phi view - ns"),
        seg2D_phi_nHits = Var(f"? hasPhi() ? phiSegment().specificRecHits().size() : 0", "int16", doc = "# hits in phi view - [0:8] range"),
        seg2D_phi_vDrift = Var(f"? hasPhi() ? phiSegment().vDrift() : {defaults.FLOAT_POS}", float, doc = "v_drift from segments with phi view"),
        seg2D_phi_normChi2 = Var(f"? hasPhi() ? (phiSegment().chi2() / phiSegment().degreesOfFreedom()) : {defaults.FLOAT_POS}", float, doc = "chi2/n.d.o.f. from segments with phi view"),
        
        seg2D_z_t0 = Var(f"? hasZed() ? zSegment().t0() : {defaults.FLOAT}", float, doc = "t0 from segments with z view - ns"),
        seg2D_z_nHits = Var(f"? hasZed() ? zSegment().specificRecHits().size() : 0", "int16", doc = "# hits in z view - [0:4] range"),
        seg2D_z_normChi2 = Var(f"? hasZed() ? (zSegment().chi2() / zSegment().degreesOfFreedom()) : {defaults.FLOAT_POS}", float, doc = "chi2/n.d.o.f. from segments with z view"),
    ),

    detIdVariables = cms.PSet(
        wheel = DetIdVar("wheel()", "int16", doc = "wheel  -  [-2:2] range"),
        sector = DetIdVar("sector()", "int16", doc = "sector - [1:14] range"
                                            "<br />sector 13 used for the second MB4 of sector 4"
                                            "<br />sector 14 used for the second MB4 of sector 10"),
        station = DetIdVar("station()", "int16", doc = "station - [1:4] range")
    ),

    globalPosVariables = cms.PSet(
        seg4D_posGlb_phi = GlobGeomVar("phi().value()", doc = "position phi in global coordinates - radians [-pi:pi]"),
        seg4D_posGlb_eta = GlobGeomVar("eta()", doc = "position eta in global coordinates"),
    ),

    globalDirVariables = cms.PSet(
        seg4D_dirGlb_phi = GlobGeomVar("phi().value()", doc = "direction phi in global coordinates - radians [-pi:pi]"),
        seg4D_dirGlb_eta = GlobGeomVar("eta()", doc = "direction eta in global coordinates"),
    )
)

from DPGAnalysis.MuonTools.muDTSegmentExtTableProducer_cfi import muDTSegmentExtTableProducer

muDTSegmentExtTable = muDTSegmentExtTableProducer.clone()

from DPGAnalysis.MuonTools.rpcRecHitFlatTableProducer_cfi import rpcRecHitFlatTableProducer

rpcRecHitFlatTable = rpcRecHitFlatTableProducer.clone(
    name = "rpcRecHit",
    src = "rpcRecHits",
    doc =  "RPC rec-hit information",

    variables = cms.PSet(
        bx = Var("BunchX()", int, doc="bunch crossing number"),
        time = Var("time()", float, doc = "time information in ns"),
        firstClusterStrip = Var("firstClusterStrip()", "int16", doc = "lowest-numbered strip in the cluster"),
        clusterSize = Var("clusterSize()", "int16", doc = "number of strips in the cluster"),
        coordX = Var("localPosition().x()", float, doc = "position x in local coordinates - cm"),
        coordY = Var("localPosition().y()", float, doc = "position y in local coordinates - cm"),
        coordZ = Var("localPosition().z()", float, doc = "position z in local coordinates - cm"),
    ),

    detIdVariables = cms.PSet(
        region = DetIdVar("region()", "int16", doc = "0: barrel, +-1: endcap"),
        ring = DetIdVar("ring()", "int16", doc = "ring id:"
                                        "<br />wheel number in barrel (from -2 to +2)"
                                        "<br />ring number in endcap (from 1 to 3)"),
        station = DetIdVar("station()", "int16", doc = "chambers at same R in barrel, chambers at same Z ion endcap"),
        layer = DetIdVar("layer()", "int16", doc = "layer id:"
                                          "<br />in station 1 and 2 for barrel, we have two layers of chambers:"
                                          "<br />layer 1 is the inner chamber and layer 2 is the outer chamber"),
        sector = DetIdVar("sector()", "int16", doc = "group of chambers at same phi"),
        subsector = DetIdVar("subsector()", "int16", doc = "Some sectors are divided along the phi direction in subsectors "
                                                  "(from 1 to 4 in Barrel, from 1 to 6 in Endcap)"),
        roll = DetIdVar("roll()", "int16", doc = "roll id (also known as eta partition):"
                                        "<br />each chamber is divided along the strip direction"),
        rawId = DetIdVar("rawId()", "uint", doc = "unique detector unit ID")
    )
)

dtrpcPointFlatTable = rpcRecHitFlatTableProducer.clone(
    name = 'dtToRpc',
    src = cms.InputTag('rpcPointProducer','RPCDTExtrapolatedPoints'),
    doc = "DT extrapolated point on RPC",

    variables = cms.PSet(
        coordX = Var("localPosition().x()", float, doc = "position x in local coordinates - cm"),
        coordY = Var("localPosition().y()", float, doc = "position y in local coordinates - cm"),
        coordZ = Var("localPosition().z()", float, doc = "position z in local coordinates - cm"),
    ),

    detIdVariables = cms.PSet(
        region = DetIdVar("region()", "int16", doc = "0: barrel, +-1: endcap"),
        ring = DetIdVar("ring()", "int16", doc = "ring id:"
                                        "<br />wheel number in barrel (from -2 to +2)"
                                        "<br />ring number in endcap (from 1 to 3)"),
        station = DetIdVar("station()", "int16", doc = "chambers at same R in barrel, chambers at same Z ion endcap"),
        layer = DetIdVar("layer()", "int16", doc = "layer id:"
                                          "<br />in station 1 and 2 for barrel, we have two layers of chambers:"
                                          "<br />layer 1 is the inner chamber and layer 2 is the outer chamber"),
        sector = DetIdVar("sector()", "int16", doc = "group of chambers at same phi"),
        subsector = DetIdVar("subsector()", "int16", doc = "Some sectors are divided along the phi direction in subsectors "
                                                  "(from 1 to 4 in Barrel, from 1 to 6 in Endcap)"),
        roll = DetIdVar("roll()", "int16", doc = "roll id (also known as eta partition):"
                                        "<br />each chamber is divided along the strip direction"),
        rawId = DetIdVar("rawId()", "uint", doc = "unique detector unit ID")
    )
)

cscrpcPointFlatTable = rpcRecHitFlatTableProducer.clone(
    name = 'cscToRpc',
    src = cms.InputTag('rpcPointProducer','RPCCSCExtrapolatedPoints'),
    doc = "CSC segment extrapolated on RPC",

    variables = cms.PSet(
        coordX = Var("localPosition().x()", float, doc = "position x in local coordinates - cm"),
        coordY = Var("localPosition().y()", float, doc = "position y in local coordinates - cm"),
        coordZ = Var("localPosition().z()", float, doc = "position z in local coordinates - cm"),
    ),

    detIdVariables = cms.PSet(
        region = DetIdVar("region()", "int16", doc = "0: barrel, +-1: endcap"),
        ring = DetIdVar("ring()", "int16", doc = "ring id:"
                                        "<br />wheel number in barrel (from -2 to +2)"
                                        "<br />ring number in endcap (from 1 to 3)"),
        station = DetIdVar("station()", "int16", doc = "chambers at same R in barrel, chambers at same Z ion endcap"),
        layer = DetIdVar("layer()", "int16", doc = "layer id:"
                                          "<br />in station 1 and 2 for barrel, we have two layers of chambers:"
                                          "<br />layer 1 is the inner chamber and layer 2 is the outer chamber"),
        sector = DetIdVar("sector()", "int16", doc = "group of chambers at same phi"),
        subsector = DetIdVar("subsector()", "int16", doc = "Some sectors are divided along the phi direction in subsectors "
                                                  "(from 1 to 4 in Barrel, from 1 to 6 in Endcap)"),
        roll = DetIdVar("roll()", "int16", doc = "roll id (also known as eta partition):"
                                        "<br />each chamber is divided along the strip direction"),
        rawId = DetIdVar("rawId()", "uint", doc = "unique detector unit ID")
    )
)

from DPGAnalysis.MuonTools.gemRecHitFlatTableProducer_cfi import gemRecHitFlatTableProducer

gemRecHitFlatTable = gemRecHitFlatTableProducer.clone(
    name = "gemRecHit",
    src = "gemRecHits",
    doc =  "GEM rec-hit information",

    variables = cms.PSet(
        bx = Var("BunchX()", int, doc="bunch crossing number"),
        clusterSize = Var("clusterSize()", "int16", doc = "number of strips in the cluster"),        loc_x = Var("localPosition().x()", float, doc = "hit position x in local coordinates - cm"),
        firstClusterStrip = Var("firstClusterStrip()", "int16", doc = "lowest-numbered strip in the cluster"),
        loc_phi = Var("localPosition().phi().value()", float, doc = "hit position phi in local coordinates - rad"),
        loc_y = Var("localPosition().y()", float, doc = "hit position y in local coordinates - cm"),
        loc_z = Var("localPosition().z()", float, doc = "hit position z in local coordinates - cm"),
    ),

    detIdVariables = cms.PSet(
        roll = DetIdVar("roll()", "int16", doc = "roll id, also known as eta partition:"
                                        "<br />(partitions numbered from 1 to 8)"),
        region = DetIdVar("region()", "int16", doc = "GE11 region where the hit is reconstructed"
                                            "<br />(int, positive endcap: +1, negative endcap: -1)"),
        chamber = DetIdVar("chamber()", "int16", doc = "GE11 superchamber where the hit is reconstructed"
                                              "<br />(chambers numbered from 0 to 35)"),
        layer = DetIdVar("layer()", "int16", doc = "GE11 layer where the hit is reconstructed"
                                          "<br />(layer1: 1, layer2: 2)")        
    ),

    globalPosVariables = cms.PSet(
        g_r = GlobGeomVar("perp()", doc = "hit position r in global coordinates - cm"),
        g_phi = GlobGeomVar("phi().value()", doc = "hit position phi in global coordinates -  radians [-pi:pi]"),
        g_x = GlobGeomVar("x()", doc = "hit position x in global coordinates - cm"),
        g_y = GlobGeomVar("y()", doc = "hit position y in global coordinates - cm"),
        g_z = GlobGeomVar("z()", doc = "hit position z in global coordinates - cm")
    )
)

from DPGAnalysis.MuonTools.gemSegmentFlatTableProducer_cfi import gemSegmentFlatTableProducer

gemSegmentFlatTable = gemSegmentFlatTableProducer.clone(
    name = "gemSegment",
    src = "gemSegments",
    doc =  "GEM segment information",

    variables = cms.PSet(
        chi2 = Var("chi2()", int, doc = "chi2 from segment fit"),
        bx = Var("bunchX()", int, doc="bunch crossing number"),
        posLoc_x = Var("localPosition().x()", float, doc = "position x in local coordinates - cm"),
        posLoc_y = Var("localPosition().y()", float, doc = "position y in local coordinates - cm"),
        posLoc_z = Var("localPosition().z()", float, doc = "position z in local coordinates - cm"),
        dirLoc_x = Var("localDirection().x()", float, doc = "direction x in local coordinates"),
        dirLoc_y = Var("localDirection().y()", float, doc = "direction y in local coordinates"),
        dirLoc_z = Var("localDirection().z()", float, doc = "direction z in local coordinates"),
    ),

    detIdVariables = cms.PSet(
        region = DetIdVar("region()", "int16", doc = "GE11 region where the hit is reconstructed"
                                            "<br />(int, positive endcap: +1, negative endcap: -1)"),
        ring = DetIdVar("ring()", "int16", doc = ""),
        station = DetIdVar("station()", "int16", doc = "GEM station <br />(always 1 for GE1/1)"),
        chamber = DetIdVar("chamber()", "int16", doc = "GE11 superchamber where the hit is reconstructed"
                                              "<br />(chambers numbered from 0 to 35)")
    ),
    
    globalPosVariables = cms.PSet(
        posGlb_x = GlobGeomVar("x()", doc = "position x in global coordinates - cm"),
        posGlb_y = GlobGeomVar("y()", doc = "position y in global coordinates - cm"),
        posGlb_z = GlobGeomVar("z()", doc = "position z in global coordinates - cm"),
        posGlb_phi = GlobGeomVar("phi().value()", doc = "position phi in global coordinates - radians [-pi:pi]"),
        posGlb_eta = GlobGeomVar("eta()", doc = "position eta in global coordinates"),
    ),

    globalDirVariables = cms.PSet(
        dirGlb_phi = GlobGeomVar("phi().value()", doc = "direction phi in global coordinates - radians [-pi:pi]"),
        dirGlb_eta = GlobGeomVar("eta()", doc = "direction eta in global coordinates"),
    )
)

muLocalRecoTables = cms.Sequence(rpcRecHitFlatTable
                                 + dtrpcPointFlatTable
                                 + cscrpcPointFlatTable
                                 + gemRecHitFlatTable
                                 + dtSegmentFlatTable
                                 + muDTSegmentExtTable
                                 + gemSegmentFlatTable
                                )
