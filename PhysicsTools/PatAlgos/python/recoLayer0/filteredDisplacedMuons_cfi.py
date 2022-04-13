import FWCore.ParameterSet.Config as cms

filteredDisplacedMuons = cms.EDProducer("DisplacedMuonFilterProducer",

    # Muon collections
    srcMuons = cms.InputTag("displacedMuons"),
    refMuons = cms.InputTag("muons"),

    FillTimingInfo = cms.bool(True),

    # Muon detector based isolation
    FillDetectorBasedIsolation = cms.bool(False),

    TrackIsoDeposits = cms.InputTag("displacedMuons:tracker"),
    JetIsoDeposits   = cms.InputTag("displacedMuons:jets"),
    EcalIsoDeposits  = cms.InputTag("displacedMuons:ecal"),
    HcalIsoDeposits  = cms.InputTag("displacedMuons:hcal"),
    HoIsoDeposits    = cms.InputTag("displacedMuons:ho"),

    # Filter
    minDxy     = cms.double( 0.1 ),
    minDz      = cms.double( 5. ),
    minDeltaR  = cms.double( 0.01 ),
    minDeltaPt = cms.double( 1.0 )
)
