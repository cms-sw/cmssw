import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var,P3Vars

hgcEERecHitsTable = cms.EDProducer("SimpleCaloRecHitFlatTableProducer",
    src = cms.InputTag("HGCalRecHit:HGCEERecHits"),
    cut = cms.string(""), 
    name = cms.string("RecHitHGCEE"),
    doc  = cms.string("RecHits in HGCAL Electromagnetic endcap"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(
        detId = Var('detid().rawId()', 'int', precision=-1, doc='detId'),
        energy = Var('energy', 'float', precision=14, doc='energy'),
        time = Var('time', 'float', precision=14, doc='hit time'),
    )
)

hgcRecHitsToSimClusters = cms.EDProducer("SimHitRecHitAssociationProducer",
    caloRecHits = cms.VInputTag("HGCalRecHit:HGCEERecHits",
        "HGCalRecHit:HGCHEFRecHits", "HGCalRecHit:HGCHEBRecHits",
    ),
    simClusters = cms.InputTag("mix:MergedCaloTruth"),
)

hgcEERecHitsToSimClusterTable = cms.EDProducer("CaloRecHitToSimClusterIndexTableProducer",
    cut = hgcEERecHitsTable.cut,
    src = hgcEERecHitsTable.src,
    objName = hgcEERecHitsTable.name,
    branchName = cms.string("SimCluster"),
    objMap = cms.InputTag("hgcRecHitsToSimClusters:HGCEERecHitsToSimClus"),
    docString = cms.string("SimCluster responsible for most sim energy in RecHit DetId")
)

hgcRecHitsToPFCands = cms.EDProducer("RecHitToPFCandAssociationProducer",
    caloRecHits = cms.VInputTag("HGCalRecHit:HGCEERecHits",
        "HGCalRecHit:HGCHEFRecHits", "HGCalRecHit:HGCHEBRecHits",
    ),
    pfCands = cms.InputTag("particleFlow"),
)

hgcEERecHitsToPFCandTable = cms.EDProducer("CaloRecHitToPFCandIndexTableProducer",
    cut = hgcEERecHitsTable.cut,
    src = hgcEERecHitsTable.src,
    objName = hgcEERecHitsTable.name,
    branchName = cms.string("PFCand"),
    objMap = cms.InputTag("hgcRecHitsToPFCands:HGCEERecHitsToPFCand"),
    docString = cms.string("PFCand with most associated energy in RecHit DetId")
)

hgcHEfrontRecHitsTable = hgcEERecHitsTable.clone()
hgcHEfrontRecHitsTable.src = "HGCalRecHit:HGCHEFRecHits"
hgcHEfrontRecHitsTable.name = "RecHitHGCHEF"

hgcHEfrontRecHitsToSimClusterTable = hgcEERecHitsToSimClusterTable.clone()
hgcHEfrontRecHitsToSimClusterTable.src = hgcHEfrontRecHitsTable.src
hgcHEfrontRecHitsToSimClusterTable.objName = hgcHEfrontRecHitsTable.name
hgcHEfrontRecHitsToSimClusterTable.objMap = "hgcRecHitsToSimClusters:HGCHEFRecHitsToSimClus"

hgcHEfrontRecHitsToPFCandTable = hgcEERecHitsToPFCandTable.clone()
hgcHEfrontRecHitsToPFCandTable.src = hgcHEfrontRecHitsTable.src
hgcHEfrontRecHitsToPFCandTable.objName = hgcHEfrontRecHitsTable.name
hgcHEfrontRecHitsToPFCandTable.objMap = "hgcRecHitsToPFCands:HGCHEFRecHitsToPFCand"

hgcHEbackRecHitsTable = hgcEERecHitsTable.clone()
hgcHEbackRecHitsTable.src = "HGCalRecHit:HGCHEBRecHits"
hgcHEbackRecHitsTable.name = "RecHitHGCHEB"

hgcHEbackRecHitsToSimClusterTable = hgcEERecHitsToSimClusterTable.clone()
hgcHEbackRecHitsToSimClusterTable.src = hgcHEbackRecHitsTable.src
hgcHEbackRecHitsToSimClusterTable.objName = hgcHEbackRecHitsTable.name
hgcHEbackRecHitsToSimClusterTable.objMap = "hgcRecHitsToSimClusters:HGCHEBRecHitsToSimClus"

hgcHEbackRecHitsToPFCandTable = hgcEERecHitsToPFCandTable.clone()
hgcHEbackRecHitsToPFCandTable.src = hgcHEbackRecHitsTable.src
hgcHEbackRecHitsToPFCandTable.objName = hgcHEbackRecHitsTable.name
hgcHEbackRecHitsToPFCandTable.objMap = "hgcRecHitsToPFCands:HGCHEBRecHitsToPFCand"

hgcEERecHitsPositionTable = cms.EDProducer("HGCRecHitPositionFromDetIDTableProducer",
    src = hgcEERecHitsTable.src,
    cut = hgcEERecHitsTable.cut, 
    name = hgcEERecHitsTable.name,
    doc  = hgcEERecHitsTable.doc,
)
#
hgcHEfrontRecHitsPositionTable = hgcEERecHitsPositionTable.clone()
hgcHEfrontRecHitsPositionTable.name = hgcHEfrontRecHitsTable.name
hgcHEfrontRecHitsPositionTable.src = hgcHEfrontRecHitsTable.src

hgcHEbackRecHitsPositionTable = hgcEERecHitsPositionTable.clone()
hgcHEbackRecHitsPositionTable.name = hgcHEbackRecHitsTable.name
hgcHEbackRecHitsPositionTable.src = hgcHEbackRecHitsTable.src

hgcRecHitsSequence = cms.Sequence(hgcEERecHitsTable+hgcHEbackRecHitsTable+hgcHEfrontRecHitsTable
                +hgcRecHitsToSimClusters
                +hgcRecHitsToPFCands
                +hgcEERecHitsToPFCandTable+hgcHEfrontRecHitsToPFCandTable+hgcHEbackRecHitsToPFCandTable
                +hgcEERecHitsPositionTable
                +hgcHEfrontRecHitsPositionTable
                +hgcHEbackRecHitsPositionTable
                +hgcEERecHitsToSimClusterTable
                +hgcHEfrontRecHitsToSimClusterTable
                +hgcHEbackRecHitsToSimClusterTable
)
