import FWCore.ParameterSet.Config as cms

# producer for alcaisotrk (HCAL isolated tracks)
from TrackingTools.TrackAssociator.default_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *

#TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackEcalRecHitCollection")
#TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackEcalRecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackHBHERecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HORecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackHORecHitCollection")

ValidationIsoTrk = cms.EDAnalyzer("ValidIsoTrkCalib",
	TrackAssociatorParameterBlock,
        hbheInput = cms.InputTag("IsoProd:IsoTrackHBHERecHitCollection"),
        hoInput = cms.InputTag("IsoProd:IsoTrackHORecHitCollection"),
#        eInput = cms.InputTag("IsoProd:IsoTrackEcalRecHitCollection"),
        HcalIsolTrackInput = cms.InputTag("IsoProd:HcalIsolatedTrackCollection"),
        trackInput = cms.InputTag("IsoProd:IsoTrackTracksCollection"),
        AxB = cms.string("Cone"),
        calibrationConeSize = cms.double(26.2),
        associationConeSize = cms.double(60),

        energyECALmip = cms.double(500.0), #take a broad range. put cuts later
	MinNTrackHitsBarrel = cms.int32(-1), # def=13. no actual cut, apply it later.
	MinNTECHitsEndcap = cms.int32(-1),  # def=11. no cut, apply it later.

        energyMinIso = cms.double(5.0),
        energyMaxIso = cms.double(1000.0),
	maxPNear = cms.double(500.0), #def=2; so no actual cut. apply it later 

	takeGenTracks = cms.untracked.bool(False),

        genTracksLabel = cms.InputTag("generalTracks"),
	genHBHE = cms.InputTag("hbhereco"),
	genHO = cms.InputTag("horeco"),
	genECAL = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),

)


