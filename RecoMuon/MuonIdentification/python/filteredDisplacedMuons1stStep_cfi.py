import FWCore.ParameterSet.Config as cms
from RecoMuon.MuonIdentification.isolation_cff import *

filteredDisplacedMuons1stStep = cms.EDProducer("DisplacedMuonFilterProducer",

    # Muon collections
    srcMuons = cms.InputTag("displacedMuons1stStep"), 
    refMuons = cms.InputTag("muons1stStep"),

    FillTimingInfo = cms.bool(True),

    # Muon detector based isolation
    FillDetectorBasedIsolation = cms.bool(True),

    TrackIsoDeposits = cms.InputTag("displacedMuons1stStep:tracker"),
    JetIsoDeposits   = cms.InputTag("displacedMuons1stStep:jets"),
    EcalIsoDeposits  = cms.InputTag("displacedMuons1stStep:ecal"),
    HcalIsoDeposits  = cms.InputTag("displacedMuons1stStep:hcal"),
    HoIsoDeposits    = cms.InputTag("displacedMuons1stStep:ho"),

    # Filter
    minDxy     = cms.double( 0.1 ),
    minDz      = cms.double( 5. ),
    minDeltaR  = cms.double( 0.01 ),
    minDeltaPt = cms.double( 1.0 )
)

