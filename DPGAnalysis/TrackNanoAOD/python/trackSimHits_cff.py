import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import CandVars,Var

trackerHitsPixelEndcapLowTofTable = cms.EDProducer("SimplePSimHitFlatTableProducer",
    src = cms.InputTag("g4SimHits:TrackerHitsPixelEndcapLowTof"),
    cut = cms.string(""), 
    name = cms.string("SimHitPixelECLowTof"),
    doc  = cms.string("Geant4 SimHits in tracker endcap"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(
        detId = Var('detUnitId', 'int', precision=-1, doc='detId'),
        energy = Var('energyLoss', 'float', precision=14, doc='energy'),
        pMag = Var('pabs', 'float', precision=14, doc='magnitude of momentum'),
        trackId = Var('trackId', 'int', precision=-1, doc='Geant4 track ID'),
        pdgId = Var('particleType', 'int', doc='PDG ID of associated track'),
    )
)

trackerHitsPixelEndcapLowTofPositionTable = cms.EDProducer("PSimHitPositionTableProducer",
    src = trackerHitsPixelEndcapLowTofTable.src,
    cut = trackerHitsPixelEndcapLowTofTable.cut, 
    name = trackerHitsPixelEndcapLowTofTable.name,
    doc  = trackerHitsPixelEndcapLowTofTable.doc,
)

trackerHitsPixelBarrelLowTofTable = trackerHitsPixelEndcapLowTofTable.clone()
trackerHitsPixelBarrelLowTofTable.src = "g4SimHits:TrackerHitsPixelBarrelLowTof"
trackerHitsPixelBarrelLowTofTable.name = "SimHitPixelLowTof"
trackerHitsPixelBarrelLowTofTable.doc = "Geant4 SimHits in pixel barrel"

trackerHitsPixelBarrelLowTofPositionTable = cms.EDProducer("PSimHitPositionTableProducer",
    src = trackerHitsPixelBarrelLowTofTable.src,
    cut = trackerHitsPixelBarrelLowTofTable.cut, 
    name = trackerHitsPixelBarrelLowTofTable.name,
    doc  = trackerHitsPixelBarrelLowTofTable.doc,
)

muonCSCHitsTable = trackerHitsPixelEndcapLowTofTable.clone()
muonCSCHitsTable.src = "g4SimHits:MuonCSCHits"
muonCSCHitsTable.name = "SimHitMuonCSC"
muonCSCHitsTable.doc = "Geant4 SimHits in Muon CSCs"

muonCSCHitsPositionTable = cms.EDProducer("PSimHitPositionTableProducer",
    src = muonCSCHitsTable.src,
    cut = muonCSCHitsTable.cut, 
    name = muonCSCHitsTable.name,
    doc  = muonCSCHitsTable.doc,
)

trackerSimHitTables = cms.Sequence(trackerHitsPixelEndcapLowTofTable+trackerHitsPixelEndcapLowTofPositionTable+
                            trackerHitsPixelBarrelLowTofTable+trackerHitsPixelBarrelLowTofPositionTable+
                            muonCSCHitsTable+muonCSCHitsPositionTable)
