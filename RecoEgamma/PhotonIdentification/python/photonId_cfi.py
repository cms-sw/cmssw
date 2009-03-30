import FWCore.ParameterSet.Config as cms

PhotonIDProd = cms.EDProducer("PhotonIDProducer",
    #required inputs
    #What collection of photons do I run on?
    photonProducer = cms.string('photons'),                              
    photonLabel = cms.string(''),
    #What labels do I use for my products?
    photonCutBasedIDLooseLabel = cms.string('PhotonCutBasedIDLoose'),
    photonCutBasedIDTightLabel = cms.string('PhotonCutBasedIDTight'),
    #What rechit collection do I use for ECAL iso?                          
    doCutBased = cms.bool(True),
    #switches, turn on quality cuts for various quantities.
    RequireFiducial = cms.bool(False),
    DoHollowConeTrackIsolationCut = cms.bool(True),
    DoSolidConeTrackIsolationCut = cms.bool(False),
    DoHollowConeNTrkCut = cms.bool(False),
    DoSolidConeNTrkCut = cms.bool(False),
    DoHadOverEMCut = cms.bool(False),
    DoEtaWidthCut = cms.bool(False),
    DoHcalTowerIsolationCut = cms.bool(True),
    DoEcalRecHitIsolationCut = cms.bool(False),
    DoEcalIsoRelativeCut = cms.bool(True),
    DoR9Cut = cms.bool(True),                               
    #LoosePhoton cuts EB  
    LoosePhotonHollowTrkEB = cms.double(9.0),
    LoosePhotonSolidTrkEB  = cms.double(999.9),
    LoosePhotonSolidNTrkEB = cms.int32(999),
    LoosePhotonHollowNTrkEB = cms.int32(999),
    LoosePhotonEtaWidthEB = cms.double(999.9),
    LoosePhotonHadOverEMEB = cms.double(999.9),
    LoosePhotonEcalRecHitIsoEB = cms.double(5.0),
    LoosePhotonEcalIsoRelativeCutEB = cms.double(0.15),                       
    LoosePhotonHcalTowerIsoEB = cms.double(5.0),
    LoosePhotonR9CutEB = cms.double(0.0),
    #TightPhoton cuts EB
    TightPhotonHollowTrkEB = cms.double(9.0),
    TightPhotonSolidTrkEB  = cms.double(999.9),
    TightPhotonSolidNTrkEB = cms.int32(999),
    TightPhotonHollowNTrkEB = cms.int32(999),
    TightPhotonEtaWidthEB = cms.double(999.9),
    TightPhotonHadOverEMEB = cms.double(999.9),
    TightPhotonEcalRecHitIsoEB = cms.double(5.0),
    TightPhotonEcalIsoRelativeCutEB = cms.double(0.15),
    TightPhotonHcalTowerIsoEB = cms.double(5.0),
    TightPhotonR9CutEB = cms.double(0.8),
    #LoosePhoton cuts EE  
    LoosePhotonHollowTrkEE = cms.double(9.0),
    LoosePhotonSolidTrkEE  = cms.double(999.9),
    LoosePhotonSolidNTrkEE = cms.int32(999),
    LoosePhotonHollowNTrkEE = cms.int32(999),
    LoosePhotonEtaWidthEE = cms.double(999.9),
    LoosePhotonHadOverEMEE = cms.double(999.9),
    LoosePhotonEcalRecHitIsoEE = cms.double(5.0),
    LoosePhotonEcalIsoRelativeCutEE = cms.double(0.2),
    LoosePhotonHcalTowerIsoEE = cms.double(5.0),
    LoosePhotonR9CutEE = cms.double(0.0),
    #TightPhoton cuts EE
    TightPhotonHollowTrkEE = cms.double(9.0),
    TightPhotonSolidTrkEE  = cms.double(999.9),
    TightPhotonSolidNTrkEE = cms.int32(999),
    TightPhotonHollowNTrkEE = cms.int32(999),
    TightPhotonEtaWidthEE = cms.double(999.9),
    TightPhotonHadOverEMEE = cms.double(999.9),
    TightPhotonEcalRecHitIsoEE = cms.double(5.0),
    TightPhotonEcalIsoRelativeCutEE = cms.double(0.2),
    TightPhotonHcalTowerIsoEE = cms.double(5.0),
    TightPhotonR9CutEE = cms.double(0.8)
)


